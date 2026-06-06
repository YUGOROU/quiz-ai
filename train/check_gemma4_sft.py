"""gemma-4-26B-A4B(MoE) を Unsloth で SFT 疎通（移行の最大リスク検証）。

Unsloth公式gemma-4ガイドの知見（2026-06）:
- **MoE の bnb 4bit QLoRA は不可**（BitsandBytes が MoE nn.Parameter の4bitを未サポート）。
  → 単純な transformers+bnb QLoRA は bf16常駐でOOM（実際に observed）。
- **解は Unsloth の bf16 LoRA**（load_in_4bit=False / load_in_16bit=True）。
  Unsloth の MoE-LoRA 最適化（全エキスパートのLoRAデルタを実体化しない）で
  **26B-A4B LoRA が >40GB ＝ 単一 A100/H100-80GB に載る**。
- chat_template は `gemma-4-thinking`（26B/31B用・思考モード）。enable_thinking で切替。

  uv run --with modal modal run train/check_gemma4_sft.py --repo google/gemma-4-26B-A4B
"""
import os

import modal

app = modal.App("quiz-gemma4-sft-check")

# Unsloth gemma-4 サポートに必要な版（faster-MoE は unsloth_zoo 同梱・要 upgrade）。
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "unsloth", "unsloth_zoo", "trl>=0.12", "transformers",
        "datasets", "huggingface_hub", "hf_transfer",
        "sentencepiece", "protobuf", "pillow",  # gemma4 processor は PIL 必須
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
)

corpus_vol = modal.Volume.from_name("quiz-corpus", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(image=image, gpu="A100-80GB",
              volumes={"/data": corpus_vol, "/hf": hf_cache},
              secrets=secrets, timeout=60 * 40)
def check(repo: str, max_steps: int, n_examples: int):
    import json
    import traceback

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    token = os.environ.get("HF_TOKEN")

    # MoE: 4bit不可 → bf16 LoRA（load_in_16bit=True）。Unsloth が MoE 最適化を自動適用。
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=repo, max_seq_length=2048,
            load_in_4bit=False, load_in_16bit=True,
            full_finetuning=False, token=token,
        )
    except Exception:  # noqa: BLE001
        print("[unsloth] FastModel.from_pretrained 失敗")
        traceback.print_exc()
        return {"repo": repo, "loaded": False}

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,    # text only
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,       # MoE expert(gate_up/down)もここで対象化
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        random_state=3407,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[unsloth] LoRA 装着 OK  trainable={n_trainable:,}")

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")

    rows = []
    with open("/data/sft_corpus_2/train.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_examples:
                break
    texts = []
    for r in rows:
        t = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False)
        if isinstance(t, str):
            t = t.removeprefix("<bos>")
        texts.append(t)
    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})

    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(
        output_dir="/tmp/gemma4_sft_smoke", max_steps=max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=1, max_length=2048,
        optim="adamw_8bit", report_to=[], dataset_text_field="text",
    )
    try:
        trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                             train_dataset=ds, args=cfg)
        trainer.train()
        print(f"[unsloth] SFT 疎通 OK  max_steps={max_steps}  trainable={n_trainable:,}")
        return {"repo": repo, "loaded": True, "trl_sft": True,
                "trainable": n_trainable}
    except Exception:  # noqa: BLE001
        print("[unsloth] SFTTrainer/train 失敗")
        traceback.print_exc()
        return {"repo": repo, "loaded": True, "trl_sft": False}


@app.local_entrypoint()
def main(repo: str = "google/gemma-4-26B-A4B", max_steps: int = 3,
         n_examples: int = 64, gpu: str = ""):
    fn = check.with_options(gpu=gpu) if gpu else check
    print("result:", fn.remote(repo=repo, max_steps=max_steps, n_examples=n_examples))
