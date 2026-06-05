"""割り込みモデル(LFM2.5-350M full-FT)の評価ハーネス — A/B/C を一括算出（Modal）。

buzz モデルは部分問題文 prefix を入力に `<think>…</think>{confidence:.2f}` を返す。
3指標は「同じ自信カーブの別の見方」:
  A) buzz位置 MAE（移行ゲート）: prefix を伸ばし confidence が閾値を超える位置 vs S-buzz の
     buzz_char の MAE（quiz-ai.md の合格基準 MAE≤8文字）。カーブが閾値を横切る位置。
  B) confidence回帰誤差: corpus-1 test の各 prefix で 予測conf vs ラベルconf の MAE/RMSE/相関。
     カーブの各点のラベルへの当てはまり（キャリブレーション）。
  C) 全文(100%)での confidence: 全文を入れた時の自信分布（情報を全部見た時の天井）。カーブ終端。

データは HF private データセット `YUGOROU/quiz-ai-corpus`（annotated_questions.jsonl /
sft_corpus_1/test.jsonl）から取得。モデルは `YUGOROU/quiz-buzz-sft-merged`（full bf16）。
prefix 書式は build_corpus1.py と完全一致（"問題文（{n}文字目まで）:\\n{prefix}"）。

実行:
  uv run --with modal modal run train/eval_buzz.py --n-questions 200 --threshold 0.5
  # 別モデル/別データを見る: --model-repo ... --dataset-repo ...
"""
import os

import modal

app = modal.App("quiz-buzz-eval")

_HERE = os.path.dirname(os.path.abspath(__file__))

# 依存は modal_sft.py と同一（unsloth が lfm2 をネイティブ patch＝LFM2.5 推論が安全）。
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unsloth", "trl>=0.12", "datasets", "huggingface_hub", "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]

# build_corpus1.py と完全一致させること（不一致は評価を無効化する）。
USER_TEMPLATE = "問題文（{n}文字目まで）:\n{prefix}"


def _parse_conf(text: str):
    """生成文の末尾（</think> 後）から confidence float を抜く。失敗で None。"""
    import re
    tail = text.split("</think>")[-1]
    m = re.search(r"(\d*\.\d+|\d+)", tail)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


@app.function(
    image=image, gpu="A10G", cpu=4.0, timeout=60 * 60,
    volumes={"/hf": hf_cache}, secrets=secrets,
)
def evaluate(model_repo: str, dataset_repo: str, n_questions: int,
             threshold: float, conf_split: str, batch_size: int,
             max_new_tokens: int, sweep_points: int):
    import json
    import math
    import re
    import random
    import statistics

    import torch
    from unsloth import FastModel
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")

    # --- モデル（full bf16・unsloth で lfm2 patch） ---
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_repo, max_seq_length=512,
        load_in_4bit=False, full_finetuning=False, dtype=None,
    )
    try:
        FastModel.for_inference(model)
    except Exception:  # noqa: BLE001
        model.eval()
    tokenizer.padding_side = "left"        # decoder 生成は left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    @torch.no_grad()
    def gen_confidences(user_contents):
        """user content のリスト → 予測 confidence のリスト（None あり）。バッチ生成。"""
        out_confs = []
        for i in range(0, len(user_contents), batch_size):
            batch = user_contents[i:i + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": c}],
                    tokenize=False, add_generation_prompt=True,
                ) for c in batch
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to("cuda")
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                txt = tokenizer.decode(row, skip_special_tokens=True)
                out_confs.append(_parse_conf(txt))
        return out_confs

    # --- データ取得（HF private dataset） ---
    test_path = hf_hub_download(dataset_repo, f"sft_corpus_1/{conf_split}.jsonl",
                                repo_type="dataset", token=token)
    annot_path = hf_hub_download(dataset_repo, "annotated_questions.jsonl",
                                 repo_type="dataset", token=token)
    test_rows = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    annot = {}
    for l in open(annot_path, encoding="utf-8"):
        q = json.loads(l)
        annot[q["qid"]] = q
    print(f"[eval] model={model_repo}  test_examples={len(test_rows)}  "
          f"annotated={len(annot)}  threshold={threshold}")

    # =========================================================
    # B) confidence 回帰誤差（corpus-1 {split} の各 prefix）
    # =========================================================
    b_prompts = [r["messages"][0]["content"] for r in test_rows]
    b_labels = [r["meta"]["confidence_label"] for r in test_rows]
    b_region = [bool(r["meta"].get("is_buzz_region")) for r in test_rows]
    b_pred = gen_confidences(b_prompts)

    pairs = [(p, l, rg) for p, l, rg in zip(b_pred, b_labels, b_region) if p is not None]
    n_parsed = len(pairs)
    abs_err = [abs(p - l) for p, l, _ in pairs]
    sq_err = [(p - l) ** 2 for p, l, _ in pairs]
    mae_b = statistics.mean(abs_err) if abs_err else float("nan")
    rmse_b = math.sqrt(statistics.mean(sq_err)) if sq_err else float("nan")
    # Pearson 相関（numpy 非依存で素実装）
    def pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return float("nan")
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    corr_b = pearson([p for p, _, _ in pairs], [l for _, l, _ in pairs])
    in_mae = [abs(p - l) for p, l, rg in pairs if rg]
    out_mae = [abs(p - l) for p, l, rg in pairs if not rg]

    # =========================================================
    # A) buzz位置 MAE（test qid の問題で dense sweep → 閾値交差）
    # C) 全文(100%)での confidence（sweep の終端 pos=L を流用）
    # =========================================================
    test_qids = {r["meta"]["qid"] for r in test_rows if r["meta"].get("qid") in annot}
    qids = sorted(test_qids)
    random.Random(3407).shuffle(qids)
    qids = qids[:n_questions]

    # 各問の dense prefix を1つの大きなバッチにまとめて投げる
    sweep_jobs = []  # (qid, pos)
    for qid in qids:
        q = annot[qid]
        L = q["question_length"]
        start = max(10, int(0.15 * L))
        if start >= L:
            start = max(1, L - 1)
        step = max(1, (L - start) // max(1, sweep_points - 1))
        positions = sorted(set(list(range(start, L, step)) + [L]))
        for pos in positions:
            sweep_jobs.append((qid, pos))
    sweep_prompts = [
        USER_TEMPLATE.format(n=pos, prefix=annot[qid]["question"][:pos])
        for qid, pos in sweep_jobs
    ]
    sweep_conf = gen_confidences(sweep_prompts)

    # qid ごとに (pos, conf) カーブを再構成
    curves = {}
    for (qid, pos), c in zip(sweep_jobs, sweep_conf):
        if c is None:
            continue
        curves.setdefault(qid, []).append((pos, c))

    c_full_confs = [sorted(curves[qid])[-1][1] for qid in qids if curves.get(qid)]  # C: 終端conf

    def buzz_errors(th):
        """与えた閾値 th でのカーブ横切り位置 vs buzz_char の誤差リストと未交差数。"""
        errs, nc = [], 0
        for qid in qids:
            pts = sorted(curves.get(qid, []))
            if not pts:
                continue
            buzz_char = annot[qid]["buzz_char"]
            L = annot[qid]["question_length"]
            pred, prev = None, None
            for pos, c in pts:
                if c >= th:
                    if prev is None:
                        pred = pos
                    else:
                        p0, c0 = prev
                        frac = (th - c0) / (c - c0) if c != c0 else 0.0
                        pred = p0 + frac * (pos - p0)
                    break
                prev = (pos, c)
            if pred is None:
                nc += 1
                pred = L
            errs.append(abs(pred - buzz_char))
        return errs, nc

    # 指定 threshold での A
    a_errs, no_cross = buzz_errors(threshold)
    mae_a = statistics.mean(a_errs) if a_errs else float("nan")
    median_a = statistics.median(a_errs) if a_errs else float("nan")
    within8 = sum(1 for e in a_errs if e <= 8) / len(a_errs) if a_errs else float("nan")

    # θ スイープ（オーケストレータは θ を調整可＝最良閾値での到達精度を見る）
    sweep_th = [0.30 + 0.05 * i for i in range(9)]   # 0.30..0.70
    th_results = []
    for th in sweep_th:
        e, nc = buzz_errors(th)
        if e:
            th_results.append((th, statistics.mean(e),
                               sum(1 for x in e if x <= 8) / len(e), nc))
    best = min(th_results, key=lambda r: r[1]) if th_results else None

    c_mean = statistics.mean(c_full_confs) if c_full_confs else float("nan")
    c_median = statistics.median(c_full_confs) if c_full_confs else float("nan")
    c_ge08 = sum(1 for c in c_full_confs if c >= 0.8) / len(c_full_confs) if c_full_confs else float("nan")
    c_min = min(c_full_confs) if c_full_confs else float("nan")

    # --- レポート ---
    print("\n================ buzz eval =================")
    print(f"[A] buzz位置 MAE（移行ゲート MAE≤8文字・θ={threshold}）:")
    print(f"    questions={len(a_errs)}  MAE={mae_a:.2f}文字  median={median_a:.2f}  "
          f"|err|≤8={within8*100:.1f}%  閾値未交差={no_cross}")
    print(f"[A'] θスイープ（オーケストレータ調整前提・最良閾値での到達精度）:")
    for th, mae, w8, nc in th_results:
        mark = "  ← best" if best and abs(th - best[0]) < 1e-9 else ""
        print(f"    θ={th:.2f}  MAE={mae:5.2f}  ≤8={w8*100:5.1f}%  未交差={nc}{mark}")
    if best:
        print(f"    → 最良 θ={best[0]:.2f} で MAE={best[1]:.2f}文字  "
              f"（gate {'PASS' if best[1] <= 8 else 'FAIL'}）")
    print(f"[B] confidence 回帰（corpus-1 {conf_split}・parsed {n_parsed}/{len(test_rows)}）:")
    print(f"    MAE={mae_b:.4f}  RMSE={rmse_b:.4f}  Pearson r={corr_b:.4f}")
    print(f"    buzz周辺 MAE={statistics.mean(in_mae):.4f}（n={len(in_mae)}） / "
          f"周辺外 MAE={statistics.mean(out_mae):.4f}（n={len(out_mae)}）"
          if in_mae and out_mae else "")
    print(f"[C] 全文(100%) confidence:")
    print(f"    mean={c_mean:.3f}  median={c_median:.3f}  min={c_min:.3f}  ≥0.8={c_ge08*100:.1f}%")
    print(f"[gate] A の MAE≤8文字 → {'PASS' if (a_errs and mae_a <= 8) else 'FAIL'}")
    print("============================================\n")

    return {
        "A": {"mae": mae_a, "median": median_a, "within8": within8,
              "no_cross": no_cross, "n": len(a_errs),
              "best_threshold": best[0] if best else None,
              "best_mae": best[1] if best else None},
        "B": {"mae": mae_b, "rmse": rmse_b, "pearson": corr_b, "parsed": n_parsed,
              "total": len(test_rows)},
        "C": {"mean": c_mean, "median": c_median, "min": c_min, "ge08": c_ge08},
    }


@app.local_entrypoint()
def main(model_repo: str = "YUGOROU/quiz-buzz-sft-merged",
         dataset_repo: str = "YUGOROU/quiz-ai-corpus",
         n_questions: int = 200, threshold: float = 0.5,
         conf_split: str = "test", batch_size: int = 64,
         max_new_tokens: int = 80, sweep_points: int = 20):
    res = evaluate.remote(
        model_repo=model_repo, dataset_repo=dataset_repo, n_questions=n_questions,
        threshold=threshold, conf_split=conf_split, batch_size=batch_size,
        max_new_tokens=max_new_tokens, sweep_points=sweep_points,
    )
    print("result:", res)
