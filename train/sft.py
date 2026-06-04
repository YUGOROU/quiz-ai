"""早押しクイズAI — 2モデル SFT（Unsloth + TRL）

割り込みモデル（LFM2.5-350M / corpus-1）とメインLLM（Qwen3.5-9B-Base / corpus-2）を
同一スクリプトで訓練する。`--target buzz|main` で設定プリセットを切り替える。

設計の正典は docs/quiz-ai.md / docs/corpus.md。要点:
- 両モデルとも assistant 出力は `<think>{reasoning}</think>{answer|confidence}` 形式（corpus 既定）。
- Base モデルなので chatml テンプレートを後付けし、`<|im_start|>assistant\n` 以降のみで損失を取る
  （train_on_responses_only）。プロンプト側はマスクして CLM ロス（corpus.md: 割り込みは CLM で開始）。
- 推論時も同じ chatml 整形を使うこと（プロンプト不一致は性能劣化の主因）。

実行環境は GPU（Modal A100 40GB / Vast.ai RTX 5090）。Mac では動かない。
Modal 経由は train/modal_sft.py、依存は train/README.md を参照。

  python train/sft.py --target buzz --data-root <corpusの親> --out-root outputs
  python train/sft.py --target main --data-root <corpusの親> --out-root outputs
"""

# ⚠ unsloth は torch/transformers より前に import する（最適化パッチのため）
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import argparse
import os

from datasets import load_dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

# chatml マーカー（Base モデルに後付け）。推論側もこの整形に合わせる。
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

# --- 設定プリセット -------------------------------------------------------
# LoRA rank/alpha は CLAUDE.md で未決のため、まずは堅実な既定値。
# buzz は 350M と小さく回帰的タスクなので軽量、main は 9B 知識タスクなので少し厚め。
CONFIGS = {
    # 割り込みモデル: 毎文字 buzz 判定。出力は <think>…</think>{confidence:.2f}
    "buzz": dict(
        base_model="LiquidAI/LFM2.5-350M-Base",
        data_subdir="sft_corpus_1",
        max_seq_length=512,          # corpus-1 実測: total char p99=174 / max=330
        load_in_4bit=False,          # 350M は小さく bf16 LoRA で十分
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        learning_rate=2e-4,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_ratio=0.05,
    ),
    # メインLLM: 部分問題文から回答。出力は <think>…</think>{answer}
    "main": dict(
        base_model="Qwen/Qwen3.5-9B-Base",
        data_subdir="sft_corpus_2",
        max_seq_length=2048,         # corpus-2 実測: total char p99=340 / max=866
        load_in_4bit=True,           # 9B を A100 40GB に載せる → QLoRA
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        learning_rate=1e-4,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
    ),
}

# LoRA 適用先。attention + MLP の標準集合。
#   - LFM2（hybrid conv+attn）: conv 層名は実機で要確認。載らなければ "all-linear" に。
#   - Qwen3.5（Gated DeltaNet + sparse MoE）: MoE expert / DeltaNet の射影名が
#     下記と異なる場合あり（メモリ main-llm-qwen35-9b-vllm 参照）。
#     エラー時は target_modules="all-linear" にフォールバックする。
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_dataset(tokenizer, data_root: str, subdir: str, split: str):
    """corpus の {split}.jsonl を読み、chatml 整形した text 列を作る。"""
    path = os.path.join(data_root, subdir, f"{split}.jsonl")
    if not os.path.exists(path):
        return None
    ds = load_dataset("json", data_files=path, split="train")

    def fmt(ex):
        # add_generation_prompt=False: assistant 応答まで含めて学習対象にする
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return ds.map(fmt, remove_columns=ds.column_names)


def train(target: str, data_root: str, out_root: str, seed: int = 3407,
          push_to_hub: str | None = None):
    if target not in CONFIGS:
        raise SystemExit(f"--target は {list(CONFIGS)} のいずれか")
    cfg = CONFIGS[target]
    set_seed(seed)

    out_dir = os.path.join(out_root, f"{target}_sft")
    print(f"[sft] target={target} base={cfg['base_model']} out={out_dir}")

    # --- モデル & トークナイザ ---
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,  # 自動（bf16 if supported）
    )

    # Base モデルに chatml テンプレートを後付け（推論側と一致させること）
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    model = FastModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # --- データ ---
    train_ds = build_dataset(tokenizer, data_root, cfg["data_subdir"], "train")
    val_ds = build_dataset(tokenizer, data_root, cfg["data_subdir"], "val")
    if train_ds is None:
        raise SystemExit(f"train.jsonl が見つからない: {data_root}/{cfg['data_subdir']}")
    print(f"[sft] train={len(train_ds)}  val={len(val_ds) if val_ds else 0}")

    # --- 訓練 ---
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        packing=False,  # response-only マスクのため packing は無効
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no",
        bf16=True,
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_cfg,
    )

    # プロンプト部をマスクし、assistant 応答のみで損失を取る
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    trainer.train()

    # LoRA アダプタを保存（重みマージは推論セットアップ側で実施）
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[sft] saved LoRA adapter -> {out_dir}")

    if push_to_hub:
        # ⚠ ライセンス: コーパス由来の重みは Private で。帰属表記は quiz-ai.md 参照
        model.push_to_hub(push_to_hub, private=True)
        tokenizer.push_to_hub(push_to_hub, private=True)
        print(f"[sft] pushed (private) -> {push_to_hub}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(CONFIGS))
    ap.add_argument("--data-root", required=True,
                    help="corpus の親ディレクトリ（例: ~/quiz-ai-corpus/corpus）")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--push-to-hub", default=None,
                    help="HF repo id（Private 推奨）。未指定ならローカル保存のみ")
    a = ap.parse_args()
    train(a.target, os.path.expanduser(a.data_root), os.path.expanduser(a.out_root),
          seed=a.seed, push_to_hub=a.push_to_hub)


if __name__ == "__main__":
    main()
