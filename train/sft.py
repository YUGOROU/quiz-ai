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

# ⚠ unsloth / trl / transformers / datasets は train() 内で遅延 import する。
#   理由: モジュールトップで import すると、Modal の add_local_python_source が
#   このファイルを Mac 上でローカル import した際に unsloth 不在で落ちる。
#   遅延させればモジュールトップは stdlib のみ＝どこでも import 可能になる。
#   遅延 import 時も「unsloth を transformers/trl より先」の順序を守ること。
import argparse
import os

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
        # バッチ較正(2026-06-04 A100 40GB): bs=8→9.93GiB, bs=32→15.7GiB（固定~8GiB+0.24GiB/sample）。
        # bs=64 で推定 ~23GiB(≈58%) と安全マージンを残しつつGPU飽和度を改善。実効batch=64。
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
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


def start_gpu_sampler(interval: float = 2.0):
    """別スレッドで nvml の GPU 利用率%・メモリ使用量を周期サンプリング。
    torch は SM 利用率を出せないため、Modal ダッシュボード相当の実測を CLI ログに残す。
    戻り値 (stop_event, samples, thread)。pynvml 不在/非GPUなら None。"""
    import threading
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:  # noqa: BLE001
        print(f"[gpu] sampler 無効（pynvml不可）: {e}")
        return None
    samples = {"util": [], "mem_gb": []}
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                samples["util"].append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                samples["mem_gb"].append(pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**3)
            except Exception:  # noqa: BLE001
                pass
            stop.wait(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop, samples, t


def report_gpu_sampler(sampler):
    """start_gpu_sampler の結果を集計してログ出力。"""
    if not sampler:
        return
    import statistics
    stop, samples, t = sampler
    stop.set()
    t.join(timeout=5)
    u, m = samples["util"], samples["mem_gb"]
    if u:
        print(f"[gpu] SM利用率% min/mean/max = {min(u)}/{statistics.mean(u):.0f}/{max(u)}  "
              f"nvmlメモリGiB mean/max = {statistics.mean(m):.2f}/{max(m):.2f}  "
              f"samples={len(u)} (interval毎)")


def build_dataset(tokenizer, data_root: str, subdir: str, split: str):
    """corpus の {split}.jsonl を読み、chatml 整形した text 列を作る。"""
    from datasets import load_dataset

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
          push_to_hub: str | None = None, hub_private: bool = True,
          max_steps: int = 0, batch_size: int = 0, grad_accum: int = 0):
    if target not in CONFIGS:
        raise SystemExit(f"--target は {list(CONFIGS)} のいずれか")
    cfg = CONFIGS[target]

    # バッチ等は CLI 上書き可（>0 で優先）。CONFIGS は baked なので再ビルド不要で調整するため。
    bs = batch_size or cfg["per_device_train_batch_size"]
    ga = grad_accum or cfg["gradient_accumulation_steps"]

    # push 指定時は訓練前に write トークンを fail-fast 検証（2h訓練後の push 失敗を防ぐ）
    if push_to_hub:
        from huggingface_hub import HfApi
        who = HfApi().whoami(token=os.environ.get("HF_TOKEN"))
        role = who.get("auth", {}).get("accessToken", {}).get("role")
        print(f"[sft] HF token: name={who.get('name')} role={role}")
        if role != "write":
            raise SystemExit("HF_TOKEN に write 権限なし。Modal Secret 'huggingface' を更新せよ")

    # ⚠ 遅延 import（unsloth を最初に＝transformers/trl へのパッチ適用のため）
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

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
    print(f"[sft] train={len(train_ds)}  val={len(val_ds) if val_ds else 0}  "
          f"per_device_bs={bs} grad_accum={ga} effective_bs={bs * ga}")

    # --- 訓練 ---
    # max_steps>0 のとき疎通run（数stepで停止・課金最小）。eval/save も省く。
    smoke = max_steps > 0
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        packing=False,  # response-only マスクのため packing は無効
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        max_steps=max_steps if smoke else -1,
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10 if not smoke else 1,
        save_strategy="no" if smoke else "epoch",
        eval_strategy="no" if smoke or val_ds is None else "epoch",
        bf16=True,
        # データ供給律速の解消（bs=32較正で GPU util 34%・CPU 1コア張り付き）。
        # Modal 側 cpu=8 と合わせ、tokenize/collate を並列化して A100 を飽和させる。
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
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

    import torch

    sampler = start_gpu_sampler(interval=2.0)  # nvml で GPU利用率%・メモリを周期計測
    stats = trainer.train()
    report_gpu_sampler(sampler)

    # GPU メモリ実測（バッチ調整の指標。allocated=テンソル実体 / reserved=確保枠）
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_resv = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[sft] peak GPU mem: allocated={peak_alloc:.2f} / reserved={peak_resv:.2f} "
          f"/ total={total:.1f} GiB  (利用率 {peak_resv / total * 100:.0f}%)")
    m = stats.metrics
    print(f"[sft] throughput: {m.get('train_samples_per_second')} samples/s  "
          f"runtime={m.get('train_runtime')}s  train_loss={m.get('train_loss')}")

    # LoRA アダプタを保存（推論はアダプタ or マージ済みのどちらでも可）
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[sft] saved LoRA adapter -> {out_dir}")

    if push_to_hub:
        # ⚠ ライセンス: コーパス由来の重み。既定 Private（quiz-ai.md 帰属表記を参照）。
        # アダプタのみ {repo}-lora / マージ済み16bit {repo}-merged の2種を上げる。
        token = os.environ.get("HF_TOKEN")  # Modal Secret 'huggingface' から注入
        lora_repo = f"{push_to_hub}-lora"
        merged_repo = f"{push_to_hub}-merged"

        model.push_to_hub(lora_repo, token=token, private=hub_private)
        tokenizer.push_to_hub(lora_repo, token=token, private=hub_private)
        print(f"[sft] pushed adapter ({'private' if hub_private else 'public'}) -> {lora_repo}")

        # マージ済み（16bit）。QLoRA(4bit)ベースは dequant してから LoRA をマージ。
        model.push_to_hub_merged(merged_repo, tokenizer, save_method="merged_16bit",
                                 token=token, private=hub_private)
        print(f"[sft] pushed merged 16bit ({'private' if hub_private else 'public'}) -> {merged_repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(CONFIGS))
    ap.add_argument("--data-root", required=True,
                    help="corpus の親ディレクトリ（例: ~/quiz-ai-corpus/corpus）")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--push-to-hub", default=None,
                    help="HF repo id プレフィックス。{repo}-lora と {repo}-merged を上げる")
    ap.add_argument("--public", action="store_true",
                    help="HF に public で上げる（既定は private）")
    ap.add_argument("--max-steps", type=int, default=0,
                    help=">0 で疎通run（数stepで停止・eval/save省略）。0で本番")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="per_device バッチ上書き（0でCONFIGS既定）")
    ap.add_argument("--grad-accum", type=int, default=0,
                    help="勾配累積上書き（0でCONFIGS既定）")
    a = ap.parse_args()
    train(a.target, os.path.expanduser(a.data_root), os.path.expanduser(a.out_root),
          seed=a.seed, push_to_hub=a.push_to_hub, hub_private=not a.public,
          max_steps=a.max_steps, batch_size=a.batch_size, grad_accum=a.grad_accum)


if __name__ == "__main__":
    main()
