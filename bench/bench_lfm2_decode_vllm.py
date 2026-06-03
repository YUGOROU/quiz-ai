# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "vllm>=0.17",
# ]
# ///
"""Phase 0b 実測② (vLLM版) — LFM2.5-350M を vLLM で回したときの decode 速度ベンチ。

transformers版（bench_lfm2_decode.py）と同じ指標を vLLM で測り、本番ランタイム候補
として比較する。vLLM の Automatic Prefix Caching(APC) はプレフィックス増分prefillと
相性が良いはずなので、逐次 buzz 模擬では前位置の KV を再利用できる想定。

注意:
- vLLM の generate はバッチAPI。1件ずつ呼ぶと scheduler overhead が乗るが、本番でも
  単発リクエストなら同様なので、その込みのレイテンシを測る（現実的）。
- enable_prefix_caching=True で、伸びていくプレフィックスの共通部分を再利用。
- 実行は CUDA GPU(RTX 5090想定)。Mac では動かない。

使い方（5090上）:
    uv run bench/bench_lfm2_decode_vllm.py
    uv run bench/bench_lfm2_decode_vllm.py --no-prefix-cache --enforce-eager
    uv run bench/bench_lfm2_decode_vllm.py --think-tokens 0,5,10,15,20 --budget-ms 100 \
        --out /tmp/lfm2_vllm.json
"""
from __future__ import annotations

import argparse
import json
import time

from common import (SYNTH_PREFIXES, print_sweep_header, print_sweep_row,
                    print_verdict, summarize_lats)


def bench_pure_decode(llm, sp_cls, n_new: int, repeats: int) -> dict:
    """安定状態の decode throughput（tok/s, per-token ms）。"""
    sp = sp_cls(max_tokens=n_new, temperature=0.0, ignore_eos=True)
    times, gens = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        outs = llm.generate([SYNTH_PREFIXES[0]], sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        g = len(outs[0].outputs[0].token_ids)
        if g > 0:
            times.append(dt / g)
            gens.append(g)
    per_tok = sum(times) / len(times) if times else float("nan")
    return {"per_token_ms": per_tok * 1e3,
            "tok_per_s": (1.0 / per_tok) if per_tok else float("nan"),
            "n_new": n_new, "repeats": repeats}


def bench_sequential_buzz(llm, sp_cls, think_tokens: int,
                          char_stride: int, max_positions: int) -> list[float]:
    """各位置で prefix[:n] から think を think_tokens 個生成する wall time(ms) を集める。
    APC 有効なら共通プレフィックスの prefill は再利用される。"""
    lats: list[float] = []
    # think_tokens=0 は最小生成(1tok)で代替（vLLMは0トークン生成不可）。回帰ヘッド相当の下限目安。
    n_gen = max(1, think_tokens)
    sp = sp_cls(max_tokens=n_gen, temperature=0.0, ignore_eos=True)
    for text in SYNTH_PREFIXES:
        L = len(text)
        start = max(4, int(L * 0.2))
        for n in range(start, min(L, max_positions) + 1, char_stride):
            prefix = text[:n]
            t0 = time.perf_counter()
            llm.generate([prefix], sp, use_tqdm=False)
            lats.append((time.perf_counter() - t0) * 1e3)
    return lats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M-Base")
    ap.add_argument("--think-tokens", default="0,5,10,15,20")
    ap.add_argument("--budget-ms", type=float, default=100.0)
    ap.add_argument("--char-stride", type=int, default=1)
    ap.add_argument("--max-positions", type=int, default=60)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-mem", type=float, default=0.30,
                    help="gpu_memory_utilization。350Mは小さいので低めでOK")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="APCを無効化（増分prefill再利用なしの比較用）")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="CUDA graph を無効化（piecewise graph との比較用）")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    think_list = [int(x) for x in args.think_tokens.split(",") if x.strip() != ""]
    print(f"[load] {args.model}  dtype={args.dtype} apc={not args.no_prefix_cache} "
          f"eager={args.enforce_eager}")
    llm = LLM(
        model=args.model, dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len,
        enable_prefix_caching=not args.no_prefix_cache,
        enforce_eager=args.enforce_eager,
    )

    # warmup
    print(f"[warmup] x{args.warmup}")
    wsp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    for _ in range(args.warmup):
        llm.generate([SYNTH_PREFIXES[0]], wsp, use_tqdm=False)

    pure = bench_pure_decode(llm, SamplingParams, n_new=256, repeats=5)
    print(f"\n[pure-decode] {pure['tok_per_s']:.0f} tok/s "
          f"(per-token {pure['per_token_ms']:.2f} ms)")

    print_sweep_header(args.budget_ms)
    results, fit_real, fit_est = {}, [], []
    for n in think_list:
        lats = bench_sequential_buzz(llm, SamplingParams, n,
                                     args.char_stride, args.max_positions)
        s = summarize_lats(lats)
        incr_est = (n + 1) * pure["per_token_ms"]
        ok = print_sweep_row(n, s, args.budget_ms, incr_est)
        results[n] = {**s, "fits_budget_p90": ok, "incremental_est_ms": incr_est}
        if ok:
            fit_real.append(n)
        if incr_est <= args.budget_ms:
            fit_est.append(n)

    print_verdict(args.budget_ms, fit_real, fit_est)

    if args.out:
        payload = {"backend": "vllm", "model": args.model, "dtype": args.dtype,
                   "apc": not args.no_prefix_cache, "enforce_eager": args.enforce_eager,
                   "budget_ms": args.budget_ms, "pure_decode": pure,
                   "sequential_buzz": results, "fit_real_p90": fit_real,
                   "fit_incremental_est": fit_est}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
