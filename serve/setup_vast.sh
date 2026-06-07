#!/usr/bin/env bash
# Phase 0b: Vast 5090 箱のブートストラップ。
# 軽量公式イメージ pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime ＋ --ssh(proxy) 前提
# （HANDOFF: vastai/pytorch 重量イメージは pull 不成立。runtime は gcc 不在）。
#
# 実行: vast に SSH 後、HF_TOKEN を export してから bash serve/setup_vast.sh
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN を export してください（read 権限可）"; exit 1
fi

echo "[setup] apt 依存（gcc=torch.compile 用 / curl=tailscale）"
apt-get update -qq
apt-get install -y -qq build-essential curl

echo "[setup] python 依存"
pip install -q -U vllm fastapi uvicorn transformers huggingface_hub

echo "[setup] モデル事前DL（gemma merged ~50GB は時間がかかる）"
export HF_HOME="${HF_HOME:-/workspace/hf}"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
for repo in ["YUGOROU/quiz-main-gemma-merged",
             "YUGOROU/quiz-buzz-reg-1.2bjp-merged"]:
    print(f"[setup] download {repo}")
    snapshot_download(repo, token=tok)
print("[setup] models ready")
PY

# --- Tailscale（任意・MacBook と接続する場合）---
# 会場の MacBook と同一 tailnet にする。WS/UDP は Tailscale 上で張る（HANDOFF 接続アーキ）。
if [ "${WITH_TAILSCALE:-0}" = "1" ]; then
  echo "[setup] tailscale 導入"
  curl -fsSL https://tailscale.com/install.sh | sh
  echo "  → 認証: tailscale up --authkey=tskey-... を手動実行（鍵はログに残さない）"
fi

echo "[setup] 完了。次:"
echo "  1) bash serve/serve_main.sh        # gemma vLLM :8000"
echo "  2) uv run serve/serve_buzz.py &    # buzz 回帰ヘッド :8001"
echo "  3) ヘルス: curl localhost:8000/v1/models ; curl localhost:8001/health"
