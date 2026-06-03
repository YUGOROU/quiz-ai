# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "tqdm",
# ]
# ///
"""Step 1: 共通前処理（S-buzz バズアノテーション）。

AI王 V2.0 train を入力に、各問題の統計的確定点 buzz_char を binary search で求め、
annotation_cache.jsonl（全件・中断再開用）と annotated_questions.jsonl（フィルタ後）を生成する。

Colab Free CPU 想定。I/Oバウンドなので ThreadPoolExecutor で並列化する。
89 t/s の単ストリーム速度は出力が極短のStep1ではほぼ無関係 → 並列度を上げて隠す。

使い方:
  # パイロット（まず200問で並列度・妥当性・所要時間を確認）
  python annotate.py --limit 200 --max-workers 32
  # 本番
  python annotate.py --max-workers 40

中断したら同じコマンドを再実行すれば cache から再開する。
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import qutils as U

# アノテーション・パラメータ（corpus.md準拠）
K = 5                 # 1ポジションあたりの試行数（--k で上書き）
THRESHOLD = 0.8       # k回中 r回以上 正答（5回中4回）
TEMPERATURE = 0.7     # 試行をばらけさせ正答率を測るため > 0
# Step1はreasoning不要（答え1語のみ）。隠れ思考の課金を止めるため none を既定に。
REASONING_EFFORT = "none"
# Crof.ai deepseek-v4-flash 価格 $/1M（in, cache, out）。cacheは通常inの約1/40。
PRICE = (0.12, 0.003, 0.21)

# 固定指示は毎コール載るため最小限に圧縮（input削減）。可変prefixは末尾。
# n（文字位置）は提示しない（位置提示によるbuzz後ずれを回避）。
PROMPT_PREFIX = "早押しクイズ（途中）。答えを1語、不明なら「不明」。\n問題: "
# フィルタ条件
MIN_QLEN = 50         # 問題文長 >= 50
MIN_ALEN = 2          # 正解の文字数 >= 2
# アノテ後フィルタ
BUZZ_RATIO_RANGE = (0.20, 0.85)


def eval_position(client, model, question: str, n: int, golds, k: int = K,
                  reasoning_effort: str | None = REASONING_EFFORT, meter=None) -> float:
    """question[:n] を与えて k 回試行し、正答率（0..1）を返す。"""
    # 固定指示を先頭に置きprefixを末尾に（プレフィックスキャッシュ整列）。
    prompt = PROMPT_PREFIX + question[:n]
    correct = 0
    for _ in range(k):
        out = U.chat(client, model, prompt, max_tokens=32, temperature=TEMPERATURE,
                     reasoning_effort=reasoning_effort, meter=meter)
        if not out or "不明" in out:
            continue
        if U.is_correct(out, golds):
            correct += 1
    return correct / k


def passes_prefilter(q: dict) -> bool:
    question = q.get("question", "")
    answers = q.get("answers", []) or []
    if len(question) < MIN_QLEN:
        return False
    if not answers or min(len(a) for a in answers) < MIN_ALEN:
        return False
    # 正解が問題文に完全一致で含まれる問題は除外
    nq = U.normalize(question)
    for a in answers:
        if U.normalize(a) and U.normalize(a) in nq:
            return False
    return True


def annotate_question(client, model, q: dict, k: int = K, threshold: float = THRESHOLD,
                      reasoning_effort: str | None = REASONING_EFFORT, meter=None) -> dict:
    """1問をアノテートして結果dictを返す（失敗時 is_valid=False）。"""
    qid = q["qid"]
    question = q["question"]
    golds = q["answers"]
    L = len(question)
    min_pos = max(10, int(L * 0.15))

    def ev(n: int) -> float:
        return eval_position(client, model, question, n, golds, k=k,
                             reasoning_effort=reasoning_effort, meter=meter)

    curve: dict[int, float] = {}

    # 全文で答えられるか（is_valid 判定）
    full_ratio = ev(L)
    curve[L] = round(full_ratio, 3)
    if full_ratio < threshold:
        return {
            "qid": qid, "question": question, "answers": golds,
            "question_length": L, "buzz_char": None, "buzz_ratio": None,
            "confidence_curve": {str(k): v for k, v in curve.items()},
            "is_valid": False,
        }

    # [min_pos, L] で「正答率 >= THRESHOLD となる最小位置」を二分探索
    lo, hi = min_pos, L
    buzz_char = L
    while lo <= hi:
        mid = (lo + hi) // 2
        ratio = curve.get(mid)
        if ratio is None:
            ratio = ev(mid)
            curve[mid] = round(ratio, 3)
        if ratio >= threshold:
            buzz_char = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return {
        "qid": qid, "question": question, "answers": golds,
        "question_length": L, "buzz_char": buzz_char,
        "buzz_ratio": round(buzz_char / L, 3),
        "confidence_curve": {str(k): curve[k] for k in sorted(curve)},
        "is_valid": True,
    }


def finalize(cache_path: str, out_path: str) -> None:
    """cache から post-filter を通った問題を annotated_questions.jsonl に書き出す。"""
    lo, hi = BUZZ_RATIO_RANGE
    n_total = n_kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in U.read_jsonl(cache_path):
            n_total += 1
            if not rec.get("is_valid"):
                continue
            br = rec.get("buzz_ratio")
            if br is None or not (lo <= br <= hi):
                continue
            import json
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1
    print(f"[finalize] {n_kept:,} / {n_total:,} 問を {out_path} に書き出し（buzz_ratio∈[{lo},{hi}], is_valid）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="corpus")
    ap.add_argument("--max-workers", type=int, default=32,
                    help="並列度。10→30→50 と上げて頭打ち点（Crof.ai同時接続上限）を探る")
    ap.add_argument("--limit", type=int, default=0, help="先頭N問のみ（0=全件）。パイロット用")
    ap.add_argument("--k", type=int, default=K,
                    help="1ポジションあたりの試行数。時間/コストが厳しければ 5→3")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="正答とみなす最小正答率。k=3なら 0.6（3回中2回）が目安")
    ap.add_argument("--reasoning-effort", default=REASONING_EFFORT,
                    help="思考量。Step1は答え1語のみ必要なので none 既定（課金停止）。"
                         "'off'/'none'/空で無効化扱い")
    args = ap.parse_args()
    reasoning_effort = args.reasoning_effort
    if reasoning_effort in ("", "off"):
        reasoning_effort = None

    cache_path = os.path.join(args.out_dir, "annotation_cache.jsonl")
    out_path = os.path.join(args.out_dir, "annotated_questions.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    client, model = U.get_client()
    meter = U.UsageMeter()
    print(f"[model] {model}  (k={args.k}, threshold={args.threshold}, "
          f"reasoning_effort={reasoning_effort!r})")

    raw = U.ensure_raw("aio_02_train.jsonl", os.path.join(args.out_dir, "raw"))
    questions = [q for q in U.read_jsonl(raw) if passes_prefilter(q)]
    if args.limit:
        questions = questions[: args.limit]

    done = U.load_done_qids(cache_path)
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[plan] prefilter後 {len(questions):,} 問 / 既処理 {len(done):,} / 今回 {len(todo):,}")

    writer = U.JsonlWriter(cache_path)
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(annotate_question, client, model, q,
                              args.k, args.threshold, reasoning_effort, meter): q
                    for q in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="annotate"):
                try:
                    writer.write(fut.result())
                except Exception as e:  # noqa: BLE001
                    q = futs[fut]
                    writer.write({"qid": q["qid"], "question": q["question"],
                                  "answers": q["answers"], "is_valid": False,
                                  "error": str(e)})
    finally:
        writer.close()

    # 本番投入前に reasoning が課金されていないか必ず確認する（パイロットの6倍超過の主因）
    # 価格は provider 依存（$/1M）。Crof.ai と Novita(HF経由) で異なる。
    ml = model.lower()
    if model == "deepseek-v4-flash":
        price = PRICE                       # Crof.ai (in, cache, out)
    elif "deepseek-v4-flash" in ml:
        price = (0.14, 0.28)                # Novita（HF Inference Providers経由）
    else:
        price = None
    print(f"[usage] {meter.summary(price)}")
    if meter.reasoning > 0:
        print("[usage][警告] reasoning_tokens>0：思考が課金されている。"
              "--reasoning-effort none が効いているか確認すること。")

    finalize(cache_path, out_path)


if __name__ == "__main__":
    main()
