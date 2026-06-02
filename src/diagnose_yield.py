# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
# ]
# ///
"""Step1 の歩留まり診断（なぜ finalize で問題が脱落するか切り分ける）。

annotation_cache.jsonl を読み、脱落を以下に分類する:
  - is_valid=False        … 全文でも正答率 < threshold（answerモデルが答えられない）
  - buzz_ratio < 0.20     … 早すぎ（簡単すぎ）で範囲外
  - buzz_ratio > 0.85     … 遅すぎ（難しすぎ）で範囲外
さらに is_valid=False 問題の「全文正答率(full_ratio)」分布を出す。
  full_ratio が 0 付近に固まる → 本当に答えられない（正当な脱落）
  full_ratio が 0.4〜0.6 に多い → kばらつき or is_correct 厳格で救える余地

--recheck: is_valid=False 問題の全文を再評価し、現行 is_correct と
  「緩和版（予測が正解の部分文字列も許容: p in ng）」で救済数を実測する。

使い方:
  uv run diagnose_yield.py --cache corpus_pilot2/annotation_cache.jsonl
  uv run diagnose_yield.py --cache corpus_pilot2/annotation_cache.jsonl --recheck --n-recheck 20
"""
from __future__ import annotations

import argparse

import qutils as U

BUZZ_RATIO_RANGE = (0.20, 0.85)
PROMPT_PREFIX = "早押しクイズ（途中）。答えを1語、不明なら「不明」。\n問題: "


def full_ratio(rec: dict) -> float | None:
    """confidence_curve から全文（最大位置=L）の正答率を取り出す。"""
    curve = rec.get("confidence_curve") or {}
    if not curve:
        return None
    L = max(int(k) for k in curve)
    return curve.get(str(L))


def is_correct_relaxed(pred: str, golds) -> bool:
    """is_correct を緩和: 完全一致・包含の双方向（ng in p または p in ng）を許容。"""
    p = U.normalize(pred)
    if not p:
        return False
    for g in golds:
        ng = U.normalize(g)
        if ng and (ng == p or ng in p or p in ng):
            return True
    return False


def bucket(fr: float | None) -> str:
    if fr is None:
        return "不明(curve無)"
    if fr <= 0.0:
        return "0.0（全く不可）"
    if fr < 0.4:
        return "(0, 0.4)"
    if fr < 0.6:
        return "[0.4, 0.6)"
    if fr < 0.8:
        return "[0.6, 0.8)"
    return "[0.8, 1.0]"


def recheck(cache_recs: list[dict], n: int, k: int = 5) -> None:
    """is_valid=False 問題を全文で再評価し is_correct 緩和の救済効果を測る。"""
    client, model = U.get_client()
    targets = [r for r in cache_recs if not r.get("is_valid")
               and r.get("question") and r.get("answers") and "error" not in r][:n]
    print(f"\n── recheck: is_valid=False {len(targets)}問を全文k={k}で再評価 ──")
    saved_strict = saved_relaxed = 0
    for r in targets:
        q, golds = r["question"], r["answers"]
        prompt = PROMPT_PREFIX + q
        cs = cr = 0
        for _ in range(k):
            out = U.chat(client, model, prompt, max_tokens=32, temperature=0.7,
                         reasoning_effort="none")
            if not out or "不明" in out:
                continue
            if U.is_correct(out, golds):
                cs += 1
            if is_correct_relaxed(out, golds):
                cr += 1
        thr = 0.8
        ps, pr = cs / k, cr / k
        if pr >= thr > ps:
            saved_relaxed += 1
            mark = "★緩和で救済"
        elif ps >= thr:
            saved_strict += 1
            mark = "（再評価で合格＝kばらつき）"
        else:
            mark = ""
        print(f"  {r['qid']} strict={ps:.1f} relaxed={pr:.1f} gold={golds} {mark}")
    print(f"  → 緩和(p in ng)で新たに救済: {saved_relaxed}問 / "
          f"再評価で合格(ばらつき): {saved_strict}問 / 対象 {len(targets)}問")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="corpus_pilot2/annotation_cache.jsonl")
    ap.add_argument("--recheck", action="store_true")
    ap.add_argument("--n-recheck", type=int, default=20)
    args = ap.parse_args()

    recs = list(U.read_jsonl(args.cache))
    if not recs:
        raise SystemExit(f"{args.cache} が空か存在しません。")

    lo, hi = BUZZ_RATIO_RANGE
    invalid = [r for r in recs if not r.get("is_valid")]
    errored = [r for r in invalid if "error" in r]
    valid = [r for r in recs if r.get("is_valid")]
    out_low = [r for r in valid if (r.get("buzz_ratio") or 0) < lo]
    out_high = [r for r in valid if (r.get("buzz_ratio") or 0) > hi]
    kept = [r for r in valid if lo <= (r.get("buzz_ratio") or -1) <= hi]

    print(f"=== 歩留まり診断: {args.cache} ===")
    print(f"総数 {len(recs)}")
    print(f"  finalize通過(kept)      : {len(kept)}  ({len(kept)/len(recs):.0%})")
    print(f"  脱落 is_valid=False     : {len(invalid)}  （内 APIエラー {len(errored)}）")
    print(f"  脱落 buzz_ratio<{lo}    : {len(out_low)}")
    print(f"  脱落 buzz_ratio>{hi}    : {len(out_high)}")

    # is_valid=False の全文正答率分布（救える余地の判定）
    from collections import Counter
    dist = Counter(bucket(full_ratio(r)) for r in invalid if "error" not in r)
    print("\n--- is_valid=False の全文正答率(full_ratio)分布 ---")
    for b in ["0.0（全く不可）", "(0, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]", "不明(curve無)"]:
        if dist.get(b):
            print(f"  {b:16}: {dist[b]}")
    print("  ※ [0.4,0.6) や [0.6,0.8) が多いほど is_correct 緩和 / k調整で救える余地が大きい")

    if args.recheck:
        recheck(recs, args.n_recheck)


if __name__ == "__main__":
    main()
