"""Modal A100 40GB で SFT を回すランチャー（無料 Starter 枠・月~14h）。

CLAUDE.md「ハードウェア・インフラ」: 訓練は Modal A100、ライブ推論は Vast.ai 5090。
コーパス（問題文を含む＝再配布禁止）は Modal Volume に private で置く。

準備:
  pip install modal && modal token new
  # コーパスを Volume にアップロード（ローカル ~/quiz-ai-corpus/corpus/ を /data 直下へ）
  modal volume create quiz-corpus
  modal volume put quiz-corpus ~/quiz-ai-corpus/corpus/sft_corpus_1 /sft_corpus_1
  modal volume put quiz-corpus ~/quiz-ai-corpus/corpus/sft_corpus_2 /sft_corpus_2

実行:
  modal run train/modal_sft.py --target buzz
  modal run train/modal_sft.py --target main
  # アダプタは Volume の /out/<target>_sft に保存され、modal volume get で回収:
  #   modal volume get quiz-corpus /out/main_sft ./outputs/main_sft
"""

import os

import modal

app = modal.App("quiz-sft")

_HERE = os.path.dirname(os.path.abspath(__file__))

# 依存解決は uv（pip より高速）。hf_transfer で重みDLを並列高速化。
# HF_HOME=/hf に向けてモデルキャッシュを Volume 永続化 → 2回目以降の run は再DLなし。
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unsloth",
        "unsloth_zoo",   # faster-MoE（gemma-4 等の MoE LoRA 最適化）
        "trl>=0.12",
        "datasets",
        "huggingface_hub",
        "hf_transfer",
        "nvidia-ml-py",  # pynvml: GPU利用率%・メモリの周期サンプリング
        "pillow",        # gemma-4 processor は PIL 必須（無いと load 時に落ちる）
        "sentencepiece",
        "protobuf",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # 並列高速DL
        "HF_HOME": "/hf",                  # モデルキャッシュは hf-cache Volume へ
    })
    # train/ を /opt/quizsft に copy=True で bake（イメージ層に確実に焼く）。
    # sft.py は重い import を遅延させてあるのでローカル import 不要。実行時に sys.path へ。
    .add_local_dir(_HERE, "/opt/quizsft", copy=True,
                   ignore=["__pycache__", "*.pyc", "*.md"])
)

SRC_DIR = "/opt/quizsft"

# コーパス Volume（/data にマウント）。出力も同じ Volume の /out へ。
corpus_vol = modal.Volume.from_name("quiz-corpus", create_if_missing=True)
# HF モデルキャッシュ Volume（初回DLのみ・以降の run で再利用しDL時間ゼロに）。
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# HF push 用 write トークン（Modal Secret 'huggingface' → 環境変数 HF_TOKEN を注入）
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image,                   # ⚠ 必須: 未指定だとデフォルト image になり unsloth/sft 不在
    gpu="A100-40GB",
    cpu=8.0,                       # 既定~1コアだと dataloader が律速し GPU利用率が頭打ち
                                   # （bs=32 較正で CPU 0.95/1.01・GPU util 34%）。8コア確保。
    timeout=60 * 60 * 12,          # 最長 12h（無料枠 ~14h/月に収める）
    volumes={"/data": corpus_vol, "/hf": hf_cache},
    secrets=secrets,
)
def run(target: str, push_to_hub: str | None = None, hub_private: bool = True,
        max_steps: int = 0, batch_size: int = 0, grad_accum: int = 0,
        epochs: int = 0, lora_r: int = 0, lora_alpha: int = 0,
        learning_rate: float = 0.0, run_tag: str = "", base_model: str = "",
        full_ft: bool = False, data_subdir: str = ""):
    import sys
    sys.path.insert(0, SRC_DIR)  # add_local_dir で bake した /opt/quizsft
    import sft  # コア訓練ロジック（sft.py）

    # Volume 直下に sft_corpus_1/ sft_corpus_2/ がある前提
    sft.train(
        target=target,
        data_root="/data",
        out_root="/data/out",
        push_to_hub=push_to_hub,
        hub_private=hub_private,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        run_tag=run_tag,
        base_model=base_model,
        full_ft=full_ft,
        data_subdir=data_subdir,
    )
    corpus_vol.commit()  # 出力を永続化
    hf_cache.commit()    # DLしたモデル重みを永続化（次回 run で再DLしない）


@app.local_entrypoint()
def main(target: str = "main", push_to_hub: str = "", public: bool = False,
         max_steps: int = 0, batch_size: int = 0, grad_accum: int = 0,
         epochs: int = 0, lora_r: int = 0, lora_alpha: int = 0,
         learning_rate: float = 0.0, run_tag: str = "", base_model: str = "",
         gpu: str = "", full_ft: bool = False, data_subdir: str = "",
         spawn: bool = False):
    # 疎通(バッチ較正): modal run train/modal_sft.py --target main --max-steps 10 --batch-size 32
    # 本番+HF push : modal run train/modal_sft.py --target main \
    #                  --push-to-hub YUGOROU/quiz-main-sft --batch-size 32
    #   → YUGOROU/quiz-main-sft-lora（アダプタ）と -merged（16bit）を private で上げる
    # バリアント例(4epoch/rank64/LR5e-5): modal run train/modal_sft.py --target main \
    #   --push-to-hub YUGOROU/quiz-main-sft-v2 --run-tag v2 \
    #   --epochs 4 --lora-r 64 --lora-alpha 64 --learning-rate 5e-5
    # CPT 再SFT例: modal run train/modal_sft.py --target main --run-tag cpt \
    #   --base-model YUGOROU/quiz-qwen-cpt-merged --push-to-hub YUGOROU/quiz-main-sft-cpt
    # 割り込み(350M)を右サイズGPUで: modal run train/modal_sft.py --target buzz \
    #   --gpu A10G --push-to-hub YUGOROU/quiz-buzz-sft
    #   ⚠ 350Mは小さくA100を飽和できない→A10G/L4で高利用率・十分速い（A100はアイドル過多）。
    # --gpu でデフォルト A100-40GB を呼び出し時オーバーライド（Function.with_options）。
    # --spawn で .spawn()＝真のサーバー側 fire-and-forget（ローカル切断で消えない。長時間run/
    #   外出時はこちら。.remote()は detached でも切断時キャンセルされ得る＝Modal警告メール）。
    fn = run.with_options(gpu=gpu) if gpu else run
    kw = dict(target=target, push_to_hub=(push_to_hub or None), hub_private=not public,
              max_steps=max_steps, batch_size=batch_size, grad_accum=grad_accum,
              epochs=epochs, lora_r=lora_r, lora_alpha=lora_alpha,
              learning_rate=learning_rate, run_tag=run_tag, base_model=base_model,
              full_ft=full_ft, data_subdir=data_subdir)
    if spawn:
        call = fn.spawn(**kw)
        print(f"[spawn] サーバー側で実行開始（Mac切断耐性）。call_id={call.object_id}")
        print("  進捗: modal app logs <app-id>（modal app list で確認）。HFに push されたら完了。")
    else:
        fn.remote(**kw)
