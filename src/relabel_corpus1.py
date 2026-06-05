# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""corpus-1 の confidence ラベルを急峻化して再ラベル（LLM不要・決定論的）。

buzz eval で判明: 標準 steepness=15/L の sigmoid は緩やかで、buzz遷移点の confidence 誤差が
大きな位置誤差(MAE 12.8文字 > ゲート8)に増幅される。steepness を上げて遷移を鋭くすれば、
同じ confidence 誤差でも位置誤差が小さくなる（位置誤差 ≒ conf誤差 / 斜面）。

think 文・meta はそのまま、**confidence の数値だけ**を新 steepness で再計算して書き換える。
LLM 再呼び出し不要・GPU不要＝数秒で完了。出力は別ディレクトリ（原本は非破壊）。

  uv run src/relabel_corpus1.py \
    --in-dir  ~/quiz-ai-corpus/corpus/sft_corpus_1 \
    --annot   ~/quiz-ai-corpus/corpus/annotated_questions.jsonl \
    --out-dir ~/quiz-ai-corpus/corpus/sft_corpus_1_steep \
    --steepness 30
"""
import argparse
import json
import math
import os


def relabel_conf(n: int, buzz_char: int, L: int, curve: dict, steepness_factor: float) -> float:
    """build_corpus1.confidence_label と同形だが steepness を可変に（既定15→例30）。"""
    steepness = steepness_factor / L
    sig = 1.0 / (1.0 + math.exp(-steepness * (n - buzz_char)))
    measured = curve.get(str(n))
    if measured is not None:
        return round(0.6 * sig + 0.4 * measured, 4)
    return round(sig, 4)


def relabel_file(in_path: str, out_path: str, annot: dict, steepness: float) -> tuple[int, int]:
    n_ok, n_skip = 0, 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            meta = item.get("meta", {})
            qid = meta.get("qid")
            n = meta.get("position")
            buzz_char = meta.get("buzz_char")
            q = annot.get(qid)
            content = item["messages"][1]["content"]
            if q is None or n is None or buzz_char is None or "</think>" not in content:
                n_skip += 1
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue
            L = q["question_length"]
            curve = q.get("confidence_curve", {})
            new_conf = relabel_conf(n, buzz_char, L, curve, steepness)
            # think 部はそのまま、末尾の confidence 数値だけ差し替え
            think_part = content[: content.rindex("</think>") + len("</think>")]
            item["messages"][1]["content"] = f"{think_part}{new_conf:.2f}"
            meta["confidence_label"] = new_conf
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_ok += 1
    return n_ok, n_skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="原 sft_corpus_1 ディレクトリ")
    ap.add_argument("--annot", required=True, help="annotated_questions.jsonl（L・curve 参照）")
    ap.add_argument("--out-dir", required=True, help="急峻化版の出力ディレクトリ")
    ap.add_argument("--steepness", type=float, default=30.0,
                    help="sigmoid steepness 係数（既定30＝標準15の2倍）")
    a = ap.parse_args()

    in_dir = os.path.expanduser(a.in_dir)
    out_dir = os.path.expanduser(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    annot = {}
    with open(os.path.expanduser(a.annot), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            annot[q["qid"]] = q
    print(f"[relabel] annotated={len(annot)}  steepness={a.steepness}/L（標準は15/L）")

    for split in ("train", "val", "test"):
        ip = os.path.join(in_dir, f"{split}.jsonl")
        if not os.path.exists(ip):
            continue
        op = os.path.join(out_dir, f"{split}.jsonl")
        ok, skip = relabel_file(ip, op, annot, a.steepness)
        print(f"[relabel] {split}: relabeled={ok}  skipped(原値維持)={skip} -> {op}")
    print(f"[relabel] done -> {out_dir}")


if __name__ == "__main__":
    main()
