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
        "trl>=0.12",
        "datasets",
        "huggingface_hub",
        "hf_transfer",
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

# HF Private モデル DL / push 用トークン（必要なら）
secrets = [modal.Secret.from_name("huggingface")] if False else []


@app.function(
    image=image,                   # ⚠ 必須: 未指定だとデフォルト image になり unsloth/sft 不在
    gpu="A100-40GB",
    timeout=60 * 60 * 12,          # 最長 12h（無料枠 ~14h/月に収める）
    volumes={"/data": corpus_vol, "/hf": hf_cache},
    secrets=secrets,
)
def run(target: str, push_to_hub: str | None = None, max_steps: int = 0):
    import sys
    sys.path.insert(0, SRC_DIR)  # add_local_dir で bake した /opt/quizsft
    import sft  # コア訓練ロジック（sft.py）

    # Volume 直下に sft_corpus_1/ sft_corpus_2/ がある前提
    sft.train(
        target=target,
        data_root="/data",
        out_root="/data/out",
        push_to_hub=push_to_hub,
        max_steps=max_steps,
    )
    corpus_vol.commit()  # 出力を永続化
    hf_cache.commit()    # DLしたモデル重みを永続化（次回 run で再DLしない）


@app.local_entrypoint()
def main(target: str = "main", push_to_hub: str = "", max_steps: int = 0):
    # 疎通: modal run train/modal_sft.py --target main --max-steps 10
    # 本番: modal run train/modal_sft.py --target main
    run.remote(target=target, push_to_hub=(push_to_hub or None), max_steps=max_steps)
