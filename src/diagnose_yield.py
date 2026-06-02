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


def _to_katakana(text: str) -> str:
    """ひらがな→カタカナに畳んでかな表記ゆれを吸収（インゲン豆 vs いんげん豆）。"""
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def _nk(text: str) -> str:
    """U.normalize + かな統一。"""
    return _to_katakana(U.normalize(text))


def match_strict(pred: str, golds) -> bool:
    """現行 qutils.is_correct（ng==p または ng in p）。"""
    return U.is_correct(pred, golds)


def match_kana(pred: str, golds) -> bool:
    """かな統一のみ追加（安全）。ng==p または ng in p を畳んだ表記で判定。"""
    p = _nk(pred)
    if not p:
        return False
    return any((ng := _nk(g)) and (ng == p or ng in p) for g in golds)


def match_kana_subset(pred: str, golds, min_len: int = 2) -> bool:
    """かな統一 + 長さガード付き部分一致（p in ng、ただし len(p)>=min_len）。
    短い予測の誤マッチ（"1"∈"1600年"）を防ぐ。"""
    if match_kana(pred, golds):
        return True
    p = _nk(pred)
    if len(p) < min_len:
        return False
    return any((ng := _nk(g)) and p in ng for g in golds)


# 比較するマッチャ（左=安全 → 右=緩い）
MATCHERS = {
    "strict": match_strict,
    "kana": match_kana,
    "kana+sub≥2": match_kana_subset,
}


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


def recheck(cache_recs: list[dict], n: int, k: int = 5, threshold: float = 0.6) -> None:
    """is_valid=False 問題を全文で再評価し、複数マッチャの救済効果と予測を実測する。"""
    client, model = U.get_client()
    targets = [r for r in cache_recs if not r.get("is_valid")
               and r.get("question") and r.get("answers") and "error" not in r][:n]
    print(f"\n── recheck: is_valid=False {len(targets)}問を全文k={k}で再評価"
          f"（合格閾値={threshold}, 予測表示あり）──")
    # マッチャごとに「strictでは不合格だが当該マッチャで合格」になった問題数を数える
    saved = {name: 0 for name in MATCHERS}
    for r in targets:
        q, golds = r["question"], r["answers"]
        prompt = PROMPT_PREFIX + q
        preds: list[str] = []
        ratios = {name: 0 for name in MATCHERS}
        for _ in range(k):
            out = U.chat(client, model, prompt, max_tokens=32, temperature=0.7,
                         reasoning_effort="none")
            if not out or "不明" in out:
                continue
            preds.append(out)
            for name, fn in MATCHERS.items():
                ratios[name] += int(fn(out, golds))
        rr = {name: ratios[name] / k for name in MATCHERS}
        strict_ok = rr["strict"] >= threshold
        marks = []
        for name in MATCHERS:
            if name != "strict" and rr[name] >= threshold > rr["strict"]:
                saved[name] += 1
                marks.append(f"★{name}で救済")
        distinct = list(dict.fromkeys(preds))[:3]
        ratio_str = " ".join(f"{n_}={rr[n_]:.1f}" for n_ in MATCHERS)
        print(f"  {r['qid']} {ratio_str} gold={golds} {' '.join(marks)}")
        print(f"      予測例: {distinct}")
    print("\n  --- 救済集計（strict不合格→各マッチャで合格になった数）---")
    for name in MATCHERS:
        if name == "strict":
            continue
        print(f"  {name:12}: +{saved[name]}問 / 対象 {len(targets)}問")
    print("  ※ kana の増分は安全に取り込める。kana+sub≥2 の追加増分は誤マッチ要注意。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="corpus_pilot2/annotation_cache.jsonl")
    ap.add_argument("--recheck", action="store_true")
    ap.add_argument("--n-recheck", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="recheckで合格とみなす最小正答率（annotate実行時の--thresholdに合わせる）")
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
        recheck(recs, args.n_recheck, threshold=args.threshold)


if __name__ == "__main__":
    main()
