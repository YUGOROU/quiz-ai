# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers>=5.0.0",
#   "accelerate",
# ]
# ///
"""Phase 0b 実測② — LFM2.5-350M の 5090 実機 decode 速度ベンチ。

割り込みモデル（LFM2.5-350M）が「毎文字 Reasoning」を char 到着間隔（~100ms想定）に
収められるかを実測する。測るのは2つ:

  (1) pure decode throughput  : 安定状態の tok/s と per-token decode 時間。
                                これだけで「think N tok の増分コスト ≈ (N+1)/tps」が出せる。
  (2) 逐次 buzz 判定の模擬     : プレフィックスを1文字ずつ伸ばし、各位置で think を N トークン
                                生成する end-to-end の wall time 分布（prefill+sampling+loop込み）。

think_tokens を {0,5,10,15,20} でスイープし、各 N の p50/p90/p99 レイテンシが
budget（既定 100ms）に収まるかを ○/× で判定する。N=0 は「回帰ヘッド相当（生成なし・
1 forward でスコア）」の下限レイテンシの目安。

※ 実行は CUDA GPU（RTX 5090 想定）。Mac/MPS では動かない（bf16/flash 非対応）。
※ ライセンス回避のため入力は合成の一般知識文（AI王コーパスは使わない）。速度測定に内容は無関係。

使い方（5090上）:
    uv run bench/bench_lfm2_decode.py
    uv run bench/bench_lfm2_decode.py --compile --attn flash_attention_2
    uv run bench/bench_lfm2_decode.py --model LiquidAI/LFM2.5-350M-Base \
        --think-tokens 0,5,10,15,20 --budget-ms 100 --out /tmp/lfm2_bench.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 速度測定用のダミー問題文（合成・一般知識／AI王コーパスではない）。
# 末尾まで1文字ずつ伸ばして「毎文字判定」を模擬する。日本語60字級を複数。
SYNTH_PREFIXES = [
    "次のうち日本で最も高い山として知られ古くから信仰の対象となり多くの芸術作品に描かれてきた標高三七七六メートルの山は何でしょう",
    "アメリカ合衆国の初代大統領を務め独立戦争では大陸軍の総司令官として活躍し現在の首都にもその名を残している人物は誰でしょう",
    "光の三原色といえば赤と緑とあと一つは何色でしょうこの三色を組み合わせることであらゆる色を表現することができます",
    "毎年十二月に発表されるその年の世相を表す漢字一字を清水寺の貫主が大きな和紙にしたためる行事が行われるのは何県でしょう",
]


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    q = sorted(xs)
    return q[min(int(p / 100 * len(q)), len(q) - 1)]


@torch.inference_mode()
def bench_pure_decode(model, tok, device, n_new: int = 256, repeats: int = 5) -> dict:
    """安定状態の decode throughput（tok/s）と per-token 時間を測る。"""
    prompt = SYNTH_PREFIXES[0]
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    times = []
    for _ in range(repeats):
        cuda_sync()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
        cuda_sync()
        dt = time.perf_counter() - t0
        gen = out.shape[1] - ids.shape[1]
        if gen > 0:
            times.append(dt / gen)  # per-token 秒
    per_tok = st.mean(times) if times else float("nan")
    return {
        "per_token_ms": per_tok * 1e3,
        "tok_per_s": (1.0 / per_tok) if per_tok else float("nan"),
        "n_new": n_new, "repeats": repeats,
    }


@torch.inference_mode()
def bench_sequential_buzz(model, tok, device, think_tokens: int,
                          char_stride: int = 1, max_positions: int = 60) -> list[float]:
    """逐次 buzz 判定の模擬。プレフィックスを char_stride 文字ずつ伸ばし、
    各位置で think を think_tokens 個生成する end-to-end wall time(ms) を集める。

    本番は KV 増分(新char分のみ前進)だが、ここでは各位置を独立に generate する
    （= prefill を毎回フル再計算する上界寄りの現実値）。増分にすれば prefill 分は
    1 token に縮むので、ここで budget に収まれば増分では更に余裕、という安全側の見方。
    """
    lats: list[float] = []
    for text in SYNTH_PREFIXES:
        full = tok(text, return_tensors="pt").input_ids.to(device)
        L = full.shape[1]
        start = max(4, int(L * 0.2))
        for n in range(start, min(L, max_positions) + 1, char_stride):
            ids = full[:, :n]
            cuda_sync()
            t0 = time.perf_counter()
            if think_tokens <= 0:
                # 回帰ヘッド相当: 生成せず 1 forward（最終 hidden が出る）だけ
                model(ids, use_cache=False)
            else:
                model.generate(ids, max_new_tokens=think_tokens, do_sample=False,
                               use_cache=True, pad_token_id=tok.eos_token_id)
            cuda_sync()
            lats.append((time.perf_counter() - t0) * 1e3)
    return lats


def summarize_lats(lats: list[float]) -> dict:
    return {
        "n": len(lats),
        "mean_ms": st.mean(lats) if lats else float("nan"),
        "p50_ms": pct(lats, 50),
        "p90_ms": pct(lats, 90),
        "p99_ms": pct(lats, 99),
        "max_ms": max(lats) if lats else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M-Base")
    ap.add_argument("--think-tokens", default="0,5,10,15,20",
                    help="thinkトークン数のスイープ（カンマ区切り）。0=回帰ヘッド相当")
    ap.add_argument("--budget-ms", type=float, default=100.0,
                    help="char到着間隔の予算。p90がこれ以下なら○")
    ap.add_argument("--char-stride", type=int, default=1)
    ap.add_argument("--max-positions", type=int, default=60)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(mode=reduce-overhead) で CUDA graph 化")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="", help="結果JSONの出力先")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU が見つかりません。本ベンチは 5090 等の CUDA 上で実行してください。")
    device = "cuda"
    gpu = torch.cuda.get_device_name(0)
    dtype = getattr(torch, args.dtype)
    think_list = [int(x) for x in args.think_tokens.split(",") if x.strip() != ""]

    print(f"[load] {args.model}  dtype={args.dtype} attn={args.attn} compile={args.compile}")
    print(f"[gpu]  {gpu}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": 0},
        attn_implementation=args.attn,
    )
    model.eval()
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception as e:  # noqa: BLE001
            print(f"[compile][警告] torch.compile 失敗、非compileで継続: {e}")

    # warmup（CUDA graph / カーネルキャッシュを温める）
    print(f"[warmup] x{args.warmup}")
    warm = tok(SYNTH_PREFIXES[0], return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model.generate(warm, max_new_tokens=16, do_sample=False,
                           use_cache=True, pad_token_id=tok.eos_token_id)
    cuda_sync()

    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    # (1) pure decode throughput
    pure = bench_pure_decode(model, tok, device)
    print(f"\n[pure-decode] {pure['tok_per_s']:.0f} tok/s "
          f"(per-token {pure['per_token_ms']:.2f} ms)")

    # (2) 逐次 buzz 模擬 + think スイープ
    print(f"\n[sequential-buzz]  budget={args.budget_ms:.0f}ms  (p90で判定)")
    print(f"  {'think_tok':>9} {'mean':>8} {'p50':>8} {'p90':>8} {'p99':>8}  判定  推定(=(N+1)/tps)")
    results = {}
    for n in think_list:
        lats = bench_sequential_buzz(model, tok, device, n,
                                     char_stride=args.char_stride,
                                     max_positions=args.max_positions)
        s = summarize_lats(lats)
        ok = "○" if s["p90_ms"] <= args.budget_ms else "×"
        est = (n + 1) * pure["per_token_ms"]  # 増分換算の理論コスト
        results[n] = {**s, "fits_budget_p90": s["p90_ms"] <= args.budget_ms,
                      "incremental_est_ms": est}
        print(f"  {n:>9} {s['mean_ms']:>7.1f} {s['p50_ms']:>7.1f} "
              f"{s['p90_ms']:>7.1f} {s['p99_ms']:>7.1f}  {ok:>3}   {est:>6.1f}ms")

    # 結論：pred/実測の両面で「budget に収まる最大 think トークン数」
    fit_real = [n for n in think_list if results[n]["fits_budget_p90"]]
    fit_est = [n for n in think_list
               if results[n]["incremental_est_ms"] <= args.budget_ms]
    print(f"\n[判定] budget {args.budget_ms:.0f}ms 内に収まる think トークン数:")
    print(f"  実測(full再計算/p90): {fit_real if fit_real else 'なし（N=0でも超過）'}")
    print(f"  増分換算(理論)       : {fit_est if fit_est else 'なし'}")
    print(f"  ※実測は prefill フル再計算の上界寄り。本番の KV 増分では増分換算に近づく。")

    payload = {
        "model": args.model, "gpu": gpu, "dtype": args.dtype, "attn": args.attn,
        "compile": args.compile, "budget_ms": args.budget_ms,
        "peak_mem_gb": mem_gb, "pure_decode": pure, "sequential_buzz": results,
        "fit_real_p90": fit_real, "fit_incremental_est": fit_est,
    }
    print(f"\n[mem] peak allocated {mem_gb:.2f} GB")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
