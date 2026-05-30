# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "tqdm",
# ]
# ///
"""Phase 0a: テキストのみのF/S評価。

annotated_questions.jsonl（パイロットの100問でOK）を使い、各問題で
オーケストレータを回して以下を集計する:
  - prefix[:65%] でのメインLLM正解率（成功基準 >= 60%）
  - buzz→回答レイテンシ（TTS除く）
  - 投機がbuzz前に完了した割合
  - キャンセル動作の健全性

使い方（Colab T4 / Mac）:
  export HF_TOKEN=hf_...            # メインLLM = gpt-oss-120b:cerebras
  python p0_run_eval.py --in corpus/annotated_questions.jsonl --limit 50

  # キャンセル機構の自己テスト（前提ズレを全問に注入）
  python p0_run_eval.py --limit 10 --reset-frac 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as stats

from tqdm import tqdm

import qutils as U
from p0_orchestrator import EpisodeConfig, run_episode
from p0_llm import MainLLM


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((len(xs) - 1) * p)))
    return xs[k]


async def main_async(args) -> None:
    questions = [q for q in U.read_jsonl(args.infile)
                 if q.get("is_valid") and q.get("buzz_char")]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit(f"有効な問題が見つかりません: {args.infile}（先に annotate.py を実行）")

    llm = MainLLM(max_tokens=args.max_tokens)
    print(f"[model] {llm.model} / 問題数 {len(questions)} / realtime={not args.fast}")

    base = EpisodeConfig(
        buzz_ratio=args.buzz_ratio, spec_lead=args.spec_lead,
        char_rate=args.char_rate, realtime=not args.fast,
    )

    results = []
    for i, q in enumerate(tqdm(questions, desc="phase0a")):
        cfg = EpisodeConfig(**vars(base))
        if args.reset_frac and (i % max(1, int(1 / args.reset_frac)) == 0):
            # buzz手前で前提ズレを注入してキャンセルを発火
            cfg.reset_at = max(1, int(len(q["question"]) * args.buzz_ratio) - 3)
        r = await run_episode(llm, q["qid"], q["question"], q["answers"], cfg, U.is_correct)
        results.append(r)

    ok = [r for r in results if r.error is None]
    errs = [r for r in results if r.error is not None]
    n_correct = sum(1 for r in ok if r.correct)
    b2a = [r.buzz_to_answer for r in ok]
    ttfts = [r.ttft for r in ok]
    totals = [r.llm_total for r in ok]
    spec_hits = sum(1 for r in ok if r.spec_done_before_buzz)
    n_cancel = sum(r.n_cancellations for r in results)

    report = {
        "model": llm.model,
        "n": len(results), "n_ok": len(ok), "n_error": len(errs),
        "accuracy": round(n_correct / len(ok), 4) if ok else 0.0,
        "n_correct": n_correct,
        "buzz_to_answer_s": {
            "mean": round(stats.mean(b2a), 3) if b2a else 0.0,
            "median": round(stats.median(b2a), 3) if b2a else 0.0,
            "p90": round(pct(b2a, 0.9), 3),
        },
        "ttft_s_mean": round(stats.mean(ttfts), 3) if ttfts else 0.0,
        "llm_total_s_mean": round(stats.mean(totals), 3) if totals else 0.0,
        "spec_done_before_buzz_rate": round(spec_hits / len(ok), 4) if ok else 0.0,
        "n_cancellations": n_cancel,
        "config": {"buzz_ratio": args.buzz_ratio, "spec_lead": args.spec_lead,
                   "char_rate": args.char_rate, "realtime": not args.fast},
    }

    print("\n=== Phase 0a レポート ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    acc = report["accuracy"]
    print(f"\n成功基準: 正解率>=0.60 → {'PASS' if acc >= 0.60 else 'FAIL'} ({acc:.0%})")
    if errs:
        print(f"⚠️ エラー {len(errs)}件 例: {errs[0].error}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"report": report,
                   "episodes": [vars(r) for r in results]},
                  f, ensure_ascii=False, indent=2)
    print(f"[saved] {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="corpus/annotated_questions.jsonl")
    ap.add_argument("--out", default="phase0a_report.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--buzz-ratio", type=float, default=0.65)
    ap.add_argument("--spec-lead", type=int, default=8, help="buzzの何文字手前で投機開始")
    ap.add_argument("--char-rate", type=float, default=12.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--fast", action="store_true", help="実時間シミュレートを省略（正解率の素早い確認用）")
    ap.add_argument("--reset-frac", type=float, default=0.0,
                    help="この割合の問題に前提ズレを注入しキャンセル機構をテスト")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
