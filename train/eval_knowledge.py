"""モデル非依存の知識天井プローブ — 任意のBaseモデルを few-shot 全文QAで測る（Modal）。

目的: Qwen3.5-9B-Base の全文QA 51%（few-shot k=4）が知識天井かを確かめるため、
別系統・同〜やや大規模のモデル（gemma4 / qwen3_5_moe / lfm2_moe / gpt_oss 等）を
**完全同一条件**（同じ few-shot 例・同じ val 評価集合・同じ採点 qutils.is_correct）で測る。

現行 `modal_infer.py --source base` は unsloth＋Qwen3.5専用でこれら新アーキを読めない。
本スクリプトは **transformers + trust_remote_code** でリポジトリ同梱 modeling code を使い、
どのアーキでもロードする（auto-class カスケードで ForCausalLM / ImageTextToText を順に試す）。

比較は `modal_infer.py --source base` と一致させること（prefix書式・header・few-shot・採点）。

実行（1モデル）:
  uv run --with modal modal run train/eval_knowledge.py \
    --repo google/gemma-4-26B-A4B --n 200 --fewshot-k 4
スモーク（ロード検証・数件）:
  uv run --with modal modal run train/eval_knowledge.py --repo <repo> --n 5
"""
import os

import modal

app = modal.App("quiz-knowledge-eval")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))  # qutils.is_correct を再利用

# 新アーキ（2026 モデル）をロードするため transformers は最新。trust_remote_code 前提で
# リポジトリ同梱の modeling code が動くよう一般的な補助依存も入れる。
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
        "einops",
        "pillow",
        "tiktoken",
        "blobfile",
        "triton",
        "bitsandbytes",
        # 注: `kernels` は入れない。最新 transformers と版が衝突し
        # `transformers.activations` の import 時点で LayerRepository が落ちて
        # 全モデルが読めなくなる（gpt-oss は kernels 無しでも transformers が
        # MXFP4 をデクオンタイズしてロードできる）。
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/hf",  # hf-cache Volume を再利用（既DLは再利用）
    })
    .add_local_dir(_SRC, "/opt/quizsrc", copy=True,
                   ignore=["__pycache__", "*.pyc"])
)

SRC_DIR = "/opt/quizsrc"

corpus_vol = modal.Volume.from_name("quiz-corpus", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image,                      # ⚠ 必須（未指定だと依存不在）
    gpu="H100",                       # 80GB。35B bf16(~70GB)まで載るよう一律。--gpu で上書き可
    volumes={"/data": corpus_vol, "/hf": hf_cache},
    secrets=secrets,
    timeout=60 * 60,
)
def run(repo: str, n: int, fewshot_k: int, seed: int, max_new_tokens: int,
        load_4bit: bool = False):
    import json
    import random
    import sys
    import traceback

    import torch

    sys.path.insert(0, SRC_DIR)
    import qutils

    token = os.environ.get("HF_TOKEN")

    # 量子化後の正解率劣化を測る用（デプロイ検定）。nf4 4bit。
    quant_cfg = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # --- モデル/トークナイザのロード（アーキ非依存・auto-class カスケード） ---
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            repo, trust_remote_code=True, token=token)
    except Exception:  # noqa: BLE001 — マルチモーダルは processor のことがある
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(
            repo, trust_remote_code=True, token=token)

    def load_model():
        from transformers import AutoModelForCausalLM
        last = None
        cands = [AutoModelForCausalLM]
        try:
            from transformers import AutoModelForImageTextToText
            cands.append(AutoModelForImageTextToText)
        except Exception:  # noqa: BLE001
            pass
        for cls in cands:
            try:
                kw = dict(torch_dtype="auto", device_map="auto",
                          trust_remote_code=True, token=token)
                if quant_cfg is not None:
                    kw["quantization_config"] = quant_cfg
                    kw.pop("torch_dtype")
                m = cls.from_pretrained(repo, **kw)
                print(f"[load] OK via {cls.__name__}  4bit={load_4bit}")
                return m
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"[load] {cls.__name__} 失敗: {type(e).__name__}: {str(e)[:160]}")
        raise last

    try:
        model = load_model()
    except Exception:  # noqa: BLE001
        print(f"[FATAL] モデルロード不能 repo={repo}")
        traceback.print_exc()
        return {"repo": repo, "loaded": False}
    model.eval()

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None and hasattr(tokenizer, "tokenizer"):
        eos_id = getattr(tokenizer.tokenizer, "eos_token_id", None)

    # --- データ: annotated_questions.jsonl（Volume）から train/val を qid 分割 ---
    path = "/data/annotated_questions.jsonl"
    train_pool, val_pool = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("is_valid"):
                continue
            (val_pool if qutils.qid_split(r["qid"]) == "val"
             else train_pool).append(r)
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    shots = train_pool[:fewshot_k]
    targets = val_pool[:n]

    # modal_infer.py --source base と完全一致の few-shot 整形
    shot_text = "".join(
        f"問題: {s['question']}\n答え: {s['answers'][0]}\n\n" for s in shots)
    header = "以下はクイズの問題と答えです。答えは簡潔に1つだけ書きます。\n\n"
    print(f"[eval] repo={repo}  few-shot k={fewshot_k}  評価{len(targets)}件")

    def raw_answer(question: str) -> str:
        prompt = header + shot_text + f"問題: {question}\n答え:"
        inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=eos_id)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        return gen.strip().splitlines()[0].strip() if gen.strip() else ""

    ok = ok_l = 0
    for i, r in enumerate(targets, 1):
        golds = r["answers"]
        try:
            pred = raw_answer(r["question"])
        except Exception as e:  # noqa: BLE001 — 1件の生成失敗で全体を落とさない
            print(f"[{i}/{len(targets)}] ⚠ gen失敗 {type(e).__name__}: {str(e)[:80]}")
            pred = ""
        c = qutils.is_correct(pred, golds)
        cl = qutils.is_correct(pred, golds, loose=True)
        ok += c
        ok_l += cl
        mark = "✅" if c else ("➕" if cl else "❌")
        print(f"[{i}/{len(targets)}] {mark} gold={golds} pred={pred!r}")

    N = len(targets) or 1
    print(f"\n[eval] === {repo}  N={N} ===")
    print(f"  全文正解率: strict {ok}/{N}={ok / N:.1%}  /  loose {ok_l}/{N}={ok_l / N:.1%}")
    return {"repo": repo, "loaded": True, "n": N,
            "strict": ok / N, "loose": ok_l / N}


@app.local_entrypoint()
def main(repo: str, n: int = 200, fewshot_k: int = 4, seed: int = 3407,
         max_new_tokens: int = 24, gpu: str = "", load_4bit: bool = False):
    fn = run.with_options(gpu=gpu) if gpu else run
    res = fn.remote(repo=repo, n=n, fewshot_k=fewshot_k, seed=seed,
                    max_new_tokens=max_new_tokens, load_4bit=load_4bit)
    print("result:", res)
