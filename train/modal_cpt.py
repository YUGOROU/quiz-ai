"""Modal A100 40GB で CPT（継続事前学習）を回すランチャー（modal_sft.py の CPT 版）。

メインLLM(Qwen3.5-9B-Base)の知識天井(全文48%)を CPT で底上げする試行（docs/HANDOFF.md）。
コーパス(cpt_corpus)は src/build_cpt_corpus.py で整形し Volume へ put しておく。

⚠ broad CPT は無料枠 14h/月 を食い潰しうる（docs/HANDOFF.md）。まず疎通でstep時間を
較正し、token-budget を 14h に収まる規模へ確定してから本番 run すること。

準備（コーパス整形も Modal 上で・ローカルDL/put不要）:
  # 疎通用の小コーパス（5M tok）を Volume /cpt_corpus へ直接生成
  uv run --with modal modal run train/modal_cpt.py::prep --token-budget 5_000_000
  # 本番は --token-budget 100_000_000（疎通でstep時間を較正してから）

実行（entrypoint が prep/main の2つあるので ::main を明示）:
  # 疎通(step時間/メモリ較正): 数stepで停止
  uv run --with modal modal run train/modal_cpt.py::main --max-steps 10 --batch-size 8
  # 本番 + HF push（{repo}-lora と {repo}-merged を private で上げる）
  uv run --with modal modal run train/modal_cpt.py::main --push-to-hub YUGOROU/quiz-qwen-cpt
  # 回収（再SFT は merged を sft.py --base-model に渡す）:
  #   再SFT: modal run train/modal_sft.py --target main --base-model YUGOROU/quiz-qwen-cpt-merged \
  #            --push-to-hub YUGOROU/quiz-main-sft-cpt --run-tag cpt
"""
import os

import modal

app = modal.App("quiz-cpt")

_HERE = os.path.dirname(os.path.abspath(__file__))

# 依存・キャッシュ戦略は modal_sft.py と同一（hf_transfer / HF_HOME=/hf 永続化）。
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unsloth",
        "trl>=0.12",
        "datasets",
        "huggingface_hub",
        "hf_transfer",
        "nvidia-ml-py",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/hf",
    })
    .add_local_dir(_HERE, "/opt/quizsft", copy=True,
                   ignore=["__pycache__", "*.pyc", "*.md"])
    # コーパス整形ロジック（src/build_cpt_corpus.py）も同じ /opt/quizsft に焼く。
    # → prep ジョブが Modal 上で wiki40b-ja を整形し Volume へ直接書ける（ローカルDL/put不要）。
    .add_local_file(os.path.join(_HERE, "..", "src", "build_cpt_corpus.py"),
                    "/opt/quizsft/build_cpt_corpus.py", copy=True)
)

SRC_DIR = "/opt/quizsft"

corpus_vol = modal.Volume.from_name("quiz-corpus", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image,                   # ⚠ 必須（modal_sft の罠と同じ）
    cpu=4.0,                       # 整形は CPU のみ（GPU不要＝無料枠14h を消費しない）
    timeout=60 * 40,
    volumes={"/data": corpus_vol, "/hf": hf_cache},
)
def prep_corpus(token_budget: int = 100_000_000, val_docs: int = 500,
                limit_scan: int = 0, pack_seq: int = 2048):
    """wiki40b-ja を Modal 上で整形し Volume の /cpt_corpus へ直接書く。

    ローカル 1.2GB DL も modal volume put も不要。DL は hf-cache Volume に永続化され、
    本番(100M)用に再走査する際も再DLなし。疎通用は token-budget を小さく（例 5_000_000）。
    pack_seq=2048（既定）で token粒度パッキング（Qwen3.5 は sample packing 無効化されるため）。
    """
    import sys
    sys.path.insert(0, SRC_DIR)
    import build_cpt_corpus

    stats = build_cpt_corpus.build(
        "/data", token_budget=token_budget, val_docs=val_docs, limit_scan=limit_scan,
        pack_seq=pack_seq,
    )
    corpus_vol.commit()   # /cpt_corpus を永続化
    hf_cache.commit()     # wiki40b-ja DL を永続化
    print(f"[prep] committed Volume /cpt_corpus: {stats}")


@app.local_entrypoint()
def prep(token_budget: int = 100_000_000, val_docs: int = 500, limit_scan: int = 0,
         pack_seq: int = 2048):
    # 疎通用（小コーパス・step時間計測はサイズ非依存なので5M tokで十分）:
    #   uv run --with modal modal run train/modal_cpt.py::prep --token-budget 5_000_000
    # 本番（100M）:
    #   uv run --with modal modal run train/modal_cpt.py::prep --token-budget 100_000_000
    prep_corpus.remote(token_budget=token_budget, val_docs=val_docs, limit_scan=limit_scan,
                       pack_seq=pack_seq)


@app.function(
    image=image,                   # ⚠ 必須: 未指定だとデフォルト image で ModuleNotFound
    gpu="H100",                    # Modalクレジット$250獲得で無料枠14h/tok$ 制約が消えた(2026-06-05)。
                                   # 指標を tok/$ → 実時間に切替。本ワークロードは248K語彙lm_headで
                                   # メモリ帯域律速＝帯域最大のGPUが最速: H100 実測2,970tok/s(L40S 1,130/
                                   # A100-80GB 1,696)。80GBで full-FT-CPT も将来選択可。merge(bf16~18GB)安全。
    cpu=8.0,
    timeout=60 * 60 * 24,          # 最長 24h（Modalクレジット$250獲得で無料枠制約消滅・2026-06-05）
    volumes={"/data": corpus_vol, "/hf": hf_cache},
    secrets=secrets,
)
def run(push_to_hub: str | None = None, hub_private: bool = True,
        max_steps: int = 0, batch_size: int = 0, grad_accum: int = 0,
        epochs: int = 0, lora_r: int = 0, lora_alpha: int = 0,
        learning_rate: float = 0.0, run_tag: str = "", embed_head: bool = True,
        grad_ckpt: str = "unsloth", full_ft: bool = False):
    import sys
    sys.path.insert(0, SRC_DIR)   # cpt.py / sft.py（GPUサンプラ流用）を import
    import cpt

    cpt.train(
        data_root="/data",         # Volume 直下に cpt_corpus/ がある前提
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
        embed_head=embed_head,
        grad_ckpt=grad_ckpt,
        full_ft=full_ft,
    )
    corpus_vol.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(push_to_hub: str = "", public: bool = False,
         max_steps: int = 0, batch_size: int = 0, grad_accum: int = 0,
         epochs: int = 0, lora_r: int = 0, lora_alpha: int = 0,
         learning_rate: float = 0.0, run_tag: str = "", no_embed_head: bool = False,
         grad_ckpt: str = "unsloth", full_ft: bool = False):
    # --full-ft で全パラメータ FT-CPT（知識注入が強い・H100 80GB前提）。
    # --no-embed-head で lm_head/embed_tokens を外す（QLoRA時のみ）。--grad-ckpt true で CPU退避を止め高速化
    run.remote(push_to_hub=(push_to_hub or None), hub_private=not public,
               max_steps=max_steps, batch_size=batch_size, grad_accum=grad_accum,
               epochs=epochs, lora_r=lora_r, lora_alpha=lora_alpha,
               learning_rate=learning_rate, run_tag=run_tag, embed_head=not no_embed_head,
               grad_ckpt=grad_ckpt, full_ft=full_ft)
