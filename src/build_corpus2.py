# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "tqdm",
# ]
# ///
"""Step 3: SFT-corpus-2（メインLLM / Qwen3.5-9B-Base 用）。

annotated_questions.jsonl から1問につき exact / relaxed の2バリアントを生成。
adaptive thinking: buzz_ratio（難易度）で think_mode（full/short/none）を割り当て、
reasoning 長を変える。assistant 形式: <think>{reasoning}</think>{answer}

正解ラベルの確実性のため、reasoning のみ LLM 生成し answer は gold を連結する。
none モードは LLM 呼び出しをスキップ（コスト削減）。

出力: corpus/sft_corpus_2/{train,val,test}.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import qutils as U

# think_mode ごとの生成予算
THINK_BUDGET = {"full": 256, "short": 96, "none": 0}


def pick_think_mode(buzz_ratio: float) -> str:
    """早buzz（簡単）は短く、遅buzz（難問）は長く。"""
    if buzz_ratio < 0.40:
        return "short"
    if buzz_ratio >= 0.65:
        return "full"
    return "full" if buzz_ratio >= 0.50 else "short"


def gen_reasoning(client, model, prefix: str, buzz_char: int, answer: str, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "short":
        instr = "手がかりから1〜2文で簡潔に推論せよ。正解そのものは書かないこと。"
    else:  # full
        instr = "プレフィックスの手がかりのみを使い、正解に至る推論を3〜5文で。正解そのものは書かないこと。"
    prompt = (
        f"早押しクイズ（{buzz_char}文字目時点）のプレフィックスと正解を渡します。\n"
        f"プレフィックス: {prefix}\n正解: {answer}\n{instr} 余分な説明不要。"
    )
    out = U.chat(client, model, prompt, max_tokens=THINK_BUDGET[mode], temperature=0.7)
    return (out or "").replace("\n", " ").strip()


def build_item(client, model, q: dict, variant: str) -> dict:
    question = q["question"]
    buzz_char = q["buzz_char"]
    answer = q["answers"][0]
    end = buzz_char if variant == "exact" else min(len(question), buzz_char + 5)
    prefix = question[:end]
    mode = pick_think_mode(q["buzz_ratio"])
    reasoning = gen_reasoning(client, model, prefix, buzz_char, answer, mode)
    return {
        "messages": [
            {"role": "user", "content": f"早押しクイズ（{buzz_char}文字目時点）:\n{prefix}"},
            {"role": "assistant", "content": f"<think>{reasoning}</think>{answer}"},
        ],
        "meta": {
            "qid": q["qid"], "buzz_char": buzz_char, "variant": variant,
            "think_mode": mode, "think_budget": THINK_BUDGET[mode],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="corpus")
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_path = os.path.join(args.out_dir, "annotated_questions.jsonl")
    cache_path = os.path.join(args.out_dir, "corpus2_cache.jsonl")
    out_dir = os.path.join(args.out_dir, "sft_corpus_2")
    os.makedirs(out_dir, exist_ok=True)

    client, model = U.get_client()
    print(f"[model] {model}")

    questions = [q for q in U.read_jsonl(in_path) if q.get("is_valid") and q.get("buzz_char")]
    if args.limit:
        questions = questions[: args.limit]

    jobs = [(q, v) for q in questions for v in ("exact", "relaxed")]
    done = U.load_done_qids(cache_path, key="key")
    todo = [(q, v) for (q, v) in jobs if f"{q['qid']}:{v}" not in done]
    print(f"[plan] {len(questions):,} 問 → {len(jobs):,} 件 / 既処理 {len(done):,} / 今回 {len(todo):,}")

    writer = U.JsonlWriter(cache_path)
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(build_item, client, model, q, v): (q, v) for (q, v) in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="corpus2"):
                q, v = futs[fut]
                try:
                    item = fut.result()
                    item["key"] = f"{q['qid']}:{v}"
                    writer.write(item)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        writer.close()

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
    print(f"[done] corpus-2 split: {counts}")


if __name__ == "__main__":
    main()
