# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tqdm",
# ]
# ///
"""Step 4: RL-corpus（GRPO用）。

AI王 V2.0 train 全量（22,335問）を対象。アノテーション成功分には
buzz_char_reference / buzz_ratio_reference を付与、失敗分は null で含める。
LLM呼び出しなし（純粋な変換）なので高速。

出力: corpus/rl_corpus/{train,val,test}.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import qutils as U


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="corpus")
    args = ap.parse_args()

    raw = U.ensure_raw("aio_02_train.jsonl", os.path.join(args.out_dir, "raw"))
    annotated_path = os.path.join(args.out_dir, "annotated_questions.jsonl")
    out_dir = os.path.join(args.out_dir, "rl_corpus")
    os.makedirs(out_dir, exist_ok=True)

    # アノテーション結果を qid -> (buzz_char, buzz_ratio) で引けるようにする
    ref: dict[str, dict] = {}
    for rec in U.read_jsonl(annotated_path):
        if rec.get("is_valid") and rec.get("buzz_char"):
            ref[rec["qid"]] = rec

    splits = {s: open(os.path.join(out_dir, f"{s}.jsonl"), "w", encoding="utf-8")
              for s in ("train", "val", "test")}
    counts = {s: 0 for s in splits}
    n_with_ref = 0

    for q in U.read_jsonl(raw):
        qid = q["qid"]
        r = ref.get(qid)
        if r:
            n_with_ref += 1
        item = {
            "prompt": q["question"],
            "answer": q["answers"][0],
            "answers": q["answers"],
            "meta": {
                "qid": qid,
                "question_length": len(q["question"]),
                "buzz_char_reference": r["buzz_char"] if r else None,
                "buzz_ratio_reference": r["buzz_ratio"] if r else None,
            },
        }
        s = U.qid_split(qid)
        splits[s].write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[s] += 1

    for f in splits.values():
        f.close()
    print(f"[done] rl_corpus split: {counts} / buzz参照あり {n_with_ref:,} 問")


if __name__ == "__main__":
    main()
