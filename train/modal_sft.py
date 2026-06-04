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

import modal

app = modal.App("quiz-sft")

# unsloth は torch/cuda 同梱版を pip 解決させる。バージョンは適宜固定推奨。
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth",
        "trl>=0.12",
        "datasets",
        "huggingface_hub",
    )
    # train/sft.py をイメージに同梱（コア訓練ロジック）
    .add_local_python_source("sft", copy=True)
)

# コーパス Volume（/data にマウント）。出力も同じ Volume の /out へ。
corpus_vol = modal.Volume.from_name("quiz-corpus", create_if_missing=True)

# HF Private モデル DL / push 用トークン（必要なら）
secrets = [modal.Secret.from_name("huggingface")] if False else []


@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 12,          # 最長 12h（無料枠 ~14h/月に収める）
    volumes={"/data": corpus_vol},
    secrets=secrets,
)
def run(target: str, push_to_hub: str | None = None):
    import sft  # イメージ同梱の train/sft.py

    # Volume 直下に sft_corpus_1/ sft_corpus_2/ がある前提
    sft.train(
        target=target,
        data_root="/data",
        out_root="/data/out",
        push_to_hub=push_to_hub,
    )
    corpus_vol.commit()  # 出力を永続化


@app.local_entrypoint()
def main(target: str = "main", push_to_hub: str = ""):
    run.remote(target=target, push_to_hub=(push_to_hub or None))
