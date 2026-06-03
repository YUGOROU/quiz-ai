"""Phase 0b 実測ハーネス共通ユーティリティ。

合成プレフィックス・分位集計・スイープ表示・budget 判定を HF版/vLLM版で共有する。
標準ライブラリのみ（計測対象の依存は各ベンチ側の inline metadata で持つ）。
"""
from __future__ import annotations

import statistics as st

# 速度測定用のダミー問題文（合成・一般知識／AI王コーパスではない）。
# 末尾まで1文字ずつ伸ばして「毎文字判定」を模擬する。速度測定に内容は無関係。
SYNTH_PREFIXES = [
    "次のうち日本で最も高い山として知られ古くから信仰の対象となり多くの芸術作品に描かれてきた標高三七七六メートルの山は何でしょう",
    "アメリカ合衆国の初代大統領を務め独立戦争では大陸軍の総司令官として活躍し現在の首都にもその名を残している人物は誰でしょう",
    "光の三原色といえば赤と緑とあと一つは何色でしょうこの三色を組み合わせることであらゆる色を表現することができます",
    "毎年十二月に発表されるその年の世相を表す漢字一字を清水寺の貫主が大きな和紙にしたためる行事が行われるのは何県でしょう",
]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    q = sorted(xs)
    return q[min(int(p / 100 * len(q)), len(q) - 1)]


def summarize_lats(lats: list[float]) -> dict:
    return {
        "n": len(lats),
        "mean_ms": st.mean(lats) if lats else float("nan"),
        "p50_ms": pct(lats, 50),
        "p90_ms": pct(lats, 90),
        "p99_ms": pct(lats, 99),
        "max_ms": max(lats) if lats else float("nan"),
    }


def print_sweep_header(budget_ms: float) -> None:
    print(f"\n[sequential-buzz]  budget={budget_ms:.0f}ms  (p90で判定)")
    print(f"  {'think_tok':>9} {'mean':>8} {'p50':>8} {'p90':>8} {'p99':>8}  判定  推定(=(N+1)/tps)")


def print_sweep_row(n: int, s: dict, budget_ms: float, incr_est_ms: float) -> bool:
    ok = s["p90_ms"] <= budget_ms
    print(f"  {n:>9} {s['mean_ms']:>7.1f} {s['p50_ms']:>7.1f} "
          f"{s['p90_ms']:>7.1f} {s['p99_ms']:>7.1f}  {'○' if ok else '×':>3}   {incr_est_ms:>6.1f}ms")
    return ok


def print_verdict(budget_ms: float, fit_real: list[int], fit_est: list[int]) -> None:
    print(f"\n[判定] budget {budget_ms:.0f}ms 内に収まる think トークン数:")
    print(f"  実測(p90)      : {fit_real if fit_real else 'なし（N=0でも超過）'}")
    print(f"  増分換算(理論) : {fit_est if fit_est else 'なし'}")
