# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "tqdm",
# ]
# ///
"""Step 2: SFT-corpus-1（割り込みモデル / LFM2.5-350M 用）。

annotated_questions.jsonl から1問あたり最大16件のprefixをサンプリングし、
各prefixに confidence ラベルと短い <think> を付与する。

出力: corpus/sft_corpus_1/{train,val,test}.jsonl
assistant 形式: <think>{short_reasoning}</think>{confidence:.2f}

割り込みモデルは速度クリティカルなので think は1〜2文に固定。
中断再開: (qid:position) 単位でキャッシュ。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import qutils as U

BUZZ_REGION = 10  # buzz_char ± この範囲を「buzz周辺」とみなす


def sample_positions(buzz_char: int, L: int) -> list[int]:
    """区分A（密）/ B（前半・疎）/ C（後半・疎）から最大16件をサンプリング。"""
    pos: set[int] = set()
    # 区分A: buzz_char ± 10 を 2文字刻み（最大10件）
    a = list(range(buzz_char - BUZZ_REGION, buzz_char + BUZZ_REGION + 1, 2))
    pos.update(a[:10] if len(a) > 10 else a)
    # 区分B: 前半・疎
    pos.update([10, int(buzz_char * 0.33), int(buzz_char * 0.66)])
    # 区分C: 後半・疎
    pos.update([int(buzz_char * 1.2), int(buzz_char * 1.5), L])
    # クリップ・範囲外除去
    return sorted(p for p in pos if 1 <= p <= L)


def confidence_label(n: int, buzz_char: int, L: int, curve: dict) -> float:
    """sigmoid を基本に、実測 confidence_curve がある位置は 0.6:0.4 でブレンド。"""
    steepness = 15.0 / L
    sig = 1.0 / (1.0 + math.exp(-steepness * (n - buzz_char)))
    measured = curve.get(str(n))
    if measured is not None:
        return round(0.6 * sig + 0.4 * measured, 4)
    return round(sig, 4)


def gen_think(client, model, prefix: str, n: int, answer: str) -> str:
    prompt = (
        f"早押しクイズの問題文プレフィックス（{n}文字目まで）と正解を渡します。\n"
        f"プレフィックス: {prefix}\n正解: {answer}\n"
        f"このプレフィックスから答えを推定する思考過程を1〜2文で。解答候補と根拠のみ。"
    )
    out = U.chat(client, model, prompt, max_tokens=96, temperature=0.7)
    return (out or "").replace("\n", " ").strip()


def build_item(client, model, q: dict, n: int) -> dict:
    question = q["question"]
    buzz_char = q["buzz_char"]
    L = q["question_length"]
    answer = q["answers"][0]
    prefix = question[:n]
    conf = confidence_label(n, buzz_char, L, q.get("confidence_curve", {}))
    think = gen_think(client, model, prefix, n, answer)
    return {
        "messages": [
            {"role": "user", "content": f"問題文（{n}文字目まで）:\n{prefix}"},
            {"role": "assistant", "content": f"<think>{think}</think>{conf:.2f}"},
        ],
        "meta": {
            "qid": q["qid"], "position": n, "buzz_char": buzz_char,
            "confidence_label": conf,
            "is_buzz_region": abs(n - buzz_char) <= BUZZ_REGION,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="corpus")
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="先頭N問のみ（パイロット）")
    args = ap.parse_args()

    in_path = os.path.join(args.out_dir, "annotated_questions.jsonl")
    cache_path = os.path.join(args.out_dir, "corpus1_cache.jsonl")
    out_dir = os.path.join(args.out_dir, "sft_corpus_1")
    os.makedirs(out_dir, exist_ok=True)

    client, model = U.get_client()
    print(f"[model] {model}")

    questions = [q for q in U.read_jsonl(in_path) if q.get("is_valid") and q.get("buzz_char")]
    if args.limit:
        questions = questions[: args.limit]

    # (qid:position) 単位で展開
    jobs = []
    for q in questions:
        for n in sample_positions(q["buzz_char"], q["question_length"]):
            jobs.append((q, n))

    done = U.load_done_qids(cache_path, key="key")
    todo = [(q, n) for (q, n) in jobs if f"{q['qid']}:{n}" not in done]
    print(f"[plan] {len(questions):,} 問 → {len(jobs):,} prefix / 既処理 {len(done):,} / 今回 {len(todo):,}")

    writer = U.JsonlWriter(cache_path)
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(build_item, client, model, q, n): (q, n) for (q, n) in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="corpus1"):
                q, n = futs[fut]
                try:
                    item = fut.result()
                    item["key"] = f"{q['qid']}:{n}"
                    writer.write(item)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        writer.close()

    # qid 単位で split 出力
    splits = {s: open(os.path.join(out_dir, f"{s}.jsonl"), "w", encoding="utf-8")
              for s in ("train", "val", "test")}
    counts = {s: 0 for s in splits}
    for rec in U.read_jsonl(cache_path):
        rec.pop("key", None)
        s = U.qid_split(rec["meta"]["qid"])
        splits[s].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[s] += 1
    for f in splits.values():
        f.close()
    print(f"[done] corpus-1 split: {counts}")


if __name__ == "__main__":
    main()
