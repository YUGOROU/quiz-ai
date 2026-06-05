"""早押しクイズAI — メインLLM CPT（継続事前学習 / Unsloth + TRL）。

背景（docs/HANDOFF.md）: Qwen3.5-9B-Base の全文正解率48%は9Bの知識ハード上限。
SFT/ハイパラ/elicitation では動かないと実証決着 → parametric知識を上げる CPT が唯一のレバー。
日本語 wiki（range3/wiki40b-ja）で継続事前学習し、CPT→再SFT→全文eval で 48%が動くか検証する。

SFT（sft.py）との違い:
  - 生テキストの素の CLM（chatml 整形なし・response-only マスクなし＝全トークンで損失）。
  - target_modules に lm_head / embed_tokens を追加（Unsloth CPT レシピ。知識/新語の獲得に必須）。
    https://unsloth.ai/docs/basics/continued-pretraining
  - embedding_learning_rate を本体 learning_rate の 2〜10x 小さく（埋め込みは壊れやすい）。
    → UnslothTrainer / UnslothTrainingArguments を使う（この2LR機構のため）。
  - packing=True（節約。マスク不要なので素直にパック可）＋ 各文書末尾に EOS。
  - 高 rank（既定 r=128）。LoRA の知識注入の弱さを rank と embed/lm_head 学習で緩和。

成果物: CPT 済み 16bit マージモデルを HF へ push（{repo}-merged）。これを sft.py の
--base-model に渡して「再SFT」する（CPT が base を底上げ → SFT は format/ドメイン適合のみ）。

実行環境は GPU（Modal A100 40GB / Vast.ai 5090）。Mac では動かない。
Modal 経由は train/modal_cpt.py、依存は train/README.md を参照。

  python train/cpt.py --data-root <corpusの親> --out-root outputs \
    --push-to-hub YUGOROU/quiz-qwen-cpt
"""
# ⚠ unsloth / trl / transformers / datasets は train() 内で遅延 import（sft.py と同理由）。
import argparse
import os

# CPT 設定。SFT(main) と土台モデル・seq長は揃え、CPT 固有値だけ変える。
CONFIG = dict(
    base_model="Qwen/Qwen3.5-9B-Base",
    data_subdir="cpt_corpus",
    max_seq_length=2048,
    load_in_4bit=True,          # 9B を A100 40GB に → QLoRA
    lora_r=128,                 # 高 rank（知識注入の容量。OOM/時間が厳しければ CLI で下げる）
    lora_alpha=32,
    lora_dropout=0.0,
    use_rslora=True,            # 高 rank の安定化（scale=alpha/sqrt(r)）
    learning_rate=5e-5,         # Unsloth CPT 推奨
    embedding_learning_rate=5e-6,  # 本体の 1/10（embed/lm_head は壊れやすい）
    num_train_epochs=1,         # CPT は通常 1 epoch（足りなければ増やす）
    # バッチはスモーク run で較正（packing=True で実効トークン密度が SFT と異なる）。
    # SFT(main) は bs=64/seq~340 で ~24GiB。CPT は packing で seq=2048 を密に詰めるため
    # 1 step の実トークンが桁違い → まず控えめ bs=8 から。
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,   # 実効 32
    warmup_ratio=0.1,                # Unsloth 公式 CPT notebook に合わせ 0.1（CPTは長めwarmup）
)

# LoRA 適用先。BASE=attn+MLP（SFT と同集合）。EMBED_HEAD=lm_head/embed_tokens（CPT追加）。
# Qwen3.5（VLM＋Gated DeltaNet+sparse MoE）で射影名が違えば "all-linear" にフォールバック。
BASE_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
EMBED_HEAD_MODULES = ["lm_head", "embed_tokens"]
# 既定は embed/lm_head 込み（Unsloth CPT レシピ）。ただし 248K 語彙の embed/lm_head 学習は
# 巨大勾配の CPU 退避を誘発しスループットを大きく落とす（実測 H100 ~3.4k tok/s）。
# Qwen3.5 は既に日本語を知っており「新言語/新トークン」ではなく事実知識の注入が目的なので、
# --no-embed-head で BASE のみ（attn+MLP LoRA）にすると退避が消え桁違いに速くなりうる。
# 知識が FFN に蓄えられる前提では BASE 高rank で足りる可能性。efficacy は eval で判定。
TARGET_MODULES = BASE_MODULES + EMBED_HEAD_MODULES


def build_dataset(tokenizer, data_root: str, subdir: str, split: str):
    """cpt_corpus/{split}.jsonl（{"text": ...}）を読み、各文書末尾に EOS を付ける。

    CPT は素の CLM なので chatml 整形しない。EOS で文書境界を明示（packing 時に
    複数文書を跨いでも区切りが学習される）。
    """
    from datasets import load_dataset

    path = os.path.join(data_root, subdir, f"{split}.jsonl")
    if not os.path.exists(path):
        return None
    ds = load_dataset("json", data_files=path, split="train")
    eos = tokenizer.eos_token or ""

    def fmt(ex):
        return {"text": ex["text"] + eos}

    return ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "text"])


def train(data_root: str, out_root: str, seed: int = 3407,
          push_to_hub: str | None = None, hub_private: bool = True,
          max_steps: int = 0, batch_size: int = 0, grad_accum: int = 0,
          epochs: int = 0, lora_r: int = 0, lora_alpha: int = 0,
          learning_rate: float = 0.0, run_tag: str = "", embed_head: bool = True,
          grad_ckpt: str = "unsloth", full_ft: bool = False):
    # ⚠ import 順序: unsloth を transformers/trl より先に（sft.py と同じ規律）
    from unsloth import FastModel, UnslothTrainer, UnslothTrainingArguments
    from transformers import set_seed

    # sft.py の GPU サンプラを流用（Modal/local とも train/ が sys.path にある前提）
    from sft import start_gpu_sampler, report_gpu_sampler

    cfg = CONFIG
    # full FT は全パラメータ更新で活性化/勾配が重い → 既定 bs を下げる（LoRA は 8）。
    default_bs = 2 if full_ft else cfg["per_device_train_batch_size"]
    bs = batch_size or default_bs
    ga = grad_accum or cfg["gradient_accumulation_steps"]
    n_epochs = epochs or cfg["num_train_epochs"]
    r = lora_r or cfg["lora_r"]
    alpha = lora_alpha or cfg["lora_alpha"]
    lr = learning_rate or cfg["learning_rate"]
    emb_lr = cfg["embedding_learning_rate"]
    # use_gradient_checkpointing: "unsloth"=smart(CPU退避あり) / True=標準(退避なし) / False=無効。
    # 80GB で VRAM 余裕がある場合は True で退避を止めると高速化しうる（退避は誤発火しがち）。
    gc = {"unsloth": "unsloth", "true": True, "false": False}.get(grad_ckpt.lower(), "unsloth")

    # push 指定時は訓練前に write トークンを fail-fast 検証（長時間 run 後の push 失敗を防ぐ）
    if push_to_hub:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN 未設定（push には write トークンが要る）")
        who = HfApi().whoami(token=token)
        perm = (who.get("auth", {}).get("accessToken", {}) or {}).get("role")
        if perm not in ("write", "admin", "fineGrained"):
            raise SystemExit(f"HF_TOKEN に write 権限がない（role={perm}）")

    set_seed(seed)

    target_modules = BASE_MODULES + (EMBED_HEAD_MODULES if embed_head else [])

    out_dir = os.path.join(out_root, f"cpt{('_' + run_tag) if run_tag else ''}")
    print(f"[cpt] base={cfg['base_model']} out={out_dir}  mode={'full-FT' if full_ft else 'QLoRA'}")
    if full_ft:
        print(f"[cpt] hparams: FULL-param FT  epochs={n_epochs} lr={lr} emb_lr={emb_lr} "
              f"grad_ckpt={gc!r} bs={bs}x{ga} (embed/lm_head は full FT で自動学習)"
              + (f"  run_tag={run_tag}" if run_tag else ""))
    else:
        print(f"[cpt] hparams: QLoRA  epochs={n_epochs} lora_r={r} lora_alpha={alpha} "
              f"lr={lr} emb_lr={emb_lr} rslora={cfg['use_rslora']} grad_ckpt={gc!r} "
              f"embed_head={embed_head} targets={target_modules}"
              + (f"  run_tag={run_tag}" if run_tag else ""))

    # --- モデル & トークナイザ ---
    if full_ft:
        # full-param FT-CPT: 全パラメータ更新（LoRA の低ランク容量限界を回避＝知識注入が強い）。
        # H100 80GB 前提・4bit 無効。grad ckpt は from_pretrained へ渡す（LoRA と経路が違う／deepwiki確認）。
        # embed_tokens/lm_head は full FT で train_embedding/train_lm_head=True が自動で立つ。
        model, tokenizer = FastModel.from_pretrained(
            model_name=cfg["base_model"],
            max_seq_length=cfg["max_seq_length"],
            load_in_4bit=False,
            full_finetuning=True,
            use_gradient_checkpointing=gc,
        )
    else:
        model, tokenizer = FastModel.from_pretrained(
            model_name=cfg["base_model"],
            max_seq_length=cfg["max_seq_length"],
            load_in_4bit=cfg["load_in_4bit"],
            full_finetuning=False,
        )
        model = FastModel.get_peft_model(
            model,
            r=r,
            lora_alpha=alpha,
            lora_dropout=cfg["lora_dropout"],
            target_modules=target_modules,   # embed_head=True で lm_head/embed_tokens 込み
            bias="none",
            use_gradient_checkpointing=gc,
            use_rslora=cfg["use_rslora"],
            random_state=seed,
        )

    # --- データ ---
    train_ds = build_dataset(tokenizer, data_root, cfg["data_subdir"], "train")
    val_ds = build_dataset(tokenizer, data_root, cfg["data_subdir"], "val")
    if train_ds is None:
        raise SystemExit(f"train.jsonl が無い: {data_root}/{cfg['data_subdir']}"
                         "（先に src/build_cpt_corpus.py で整形 → Volume へ put）")
    print(f"[cpt] train={len(train_ds)} docs  val={len(val_ds) if val_ds else 0}  "
          f"per_device_bs={bs} grad_accum={ga} effective_bs={bs * ga}")

    # --- 訓練 ---
    smoke = max_steps > 0
    args = UnslothTrainingArguments(
        output_dir=out_dir,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        packing=True,                 # CPT はマスク不要 → 素直にパックして A100 を飽和
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        max_steps=max_steps if smoke else -1,
        num_train_epochs=n_epochs,
        learning_rate=lr,
        embedding_learning_rate=emb_lr,   # ← lm_head/embed_tokens 用の小さい LR
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.0,            # Unsloth 公式 CPT notebook に合わせ 0.0
        logging_steps=10 if not smoke else 1,
        save_strategy="no" if smoke else "epoch",
        eval_strategy="no" if smoke or val_ds is None else "epoch",
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
        report_to="none",
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
    )

    import torch

    sampler = start_gpu_sampler(interval=2.0)
    stats = trainer.train()
    report_gpu_sampler(sampler)

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_resv = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[cpt] peak GPU mem: allocated={peak_alloc:.2f} / reserved={peak_resv:.2f} "
          f"/ total={total:.1f} GiB  (利用率 {peak_resv / total * 100:.0f}%)")
    m = stats.metrics
    print(f"[cpt] throughput: {m.get('train_samples_per_second')} samples/s  "
          f"runtime={m.get('train_runtime')}s  train_loss={m.get('train_loss')}")

    # ローカル保存（full FT=全重み / QLoRA=アダプタ）
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[cpt] saved {'full model' if full_ft else 'LoRA adapter'} -> {out_dir}")

    if push_to_hub:
        token = os.environ.get("HF_TOKEN")
        merged_repo = f"{push_to_hub}-merged"   # 再SFT の --base-model に渡す土台

        if full_ft:
            # 非PEFT: native push_to_hub（push_to_hub_merged は PEFT 専用で警告/エラー／deepwiki確認）。
            # bf16 全重みがそのまま上がる＝再SFT 土台。-merged 名で LoRA 経路と downstream を統一。
            model.push_to_hub(merged_repo, token=token, private=hub_private)
            tokenizer.push_to_hub(merged_repo, token=token, private=hub_private)
            print(f"[cpt] pushed full 16bit model -> {merged_repo}")
        else:
            lora_repo = f"{push_to_hub}-lora"
            model.push_to_hub(lora_repo, token=token, private=hub_private)
            tokenizer.push_to_hub(lora_repo, token=token, private=hub_private)
            print(f"[cpt] pushed adapter -> {lora_repo}")
            # マージ済み 16bit。これを sft.py --base-model に渡して「再SFT」する。
            model.push_to_hub_merged(merged_repo, tokenizer, save_method="merged_16bit",
                                     token=token, private=hub_private)
        print(f"[cpt] merged base -> {merged_repo}  "
              "（再SFT: python train/sft.py --target main --base-model "
              f"{merged_repo} ...）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="corpus の親（cpt_corpus/ を含む）")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--push-to-hub", default=None,
                    help="HF repo id プレフィックス。{repo}-lora と {repo}-merged を上げる")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0,
                    help=">0 で疎通run（数stepで停止・eval/save省略）。0で本番")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=0.0)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--no-embed-head", action="store_true",
                    help="lm_head/embed_tokens を学習対象から外す（attn+MLP のみ）")
    ap.add_argument("--grad-ckpt", default="unsloth", choices=["unsloth", "true", "false"],
                    help="勾配チェックポイント。unsloth=smart(CPU退避) / true=標準(退避なし・80GBで高速) / false=無効")
    ap.add_argument("--full-ft", action="store_true",
                    help="全パラメータ FT-CPT（4bit/LoRA を使わず全重み更新。知識注入が強い・H100 80GB前提）")
    a = ap.parse_args()
    train(os.path.expanduser(a.data_root), os.path.expanduser(a.out_root),
          seed=a.seed, push_to_hub=a.push_to_hub, hub_private=not a.public,
          max_steps=a.max_steps, batch_size=a.batch_size, grad_accum=a.grad_accum,
          epochs=a.epochs, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
          learning_rate=a.learning_rate, run_tag=a.run_tag, embed_head=not a.no_embed_head,
          grad_ckpt=a.grad_ckpt, full_ft=a.full_ft)


if __name__ == "__main__":
    main()
