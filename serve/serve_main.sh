#!/usr/bin/env bash
# Phase 0b: メインLLM = gemma-4-26B-A4B SFT の vLLM serve（Vast 5090 上）。
#
# 重要な罠（HANDOFF 2026-06-07）:
#   - gemma-4 の assistant ターン終端は <turn|>(id 106)。<eos>(1) だけでは止まらず
#     生成が暴走する → stop-token-ids に 1,106 を必ず含める（レイテンシ予算に致命的）。
#   - FP8 MoE は deep_gemm が必要。pip 単体ビルドは失敗するため、FP8 を使う場合は
#     vLLM 公式 Docker イメージ（deep_gemm 同梱）で起動する。ここでは bf16/4bit を既定とし
#     FP8 を避ける（5090 32GB なら 4bit で gemma + buzz が同居可）。
#   - runtime イメージは gcc 不在で torch.compile が落ちる → setup_vast.sh で build-essential。
#
# 使い方:
#   export HF_TOKEN=hf_...
#   bash serve/serve_main.sh
set -euo pipefail

MODEL="${MAIN_MODEL:-YUGOROU/quiz-main-gemma-merged}"
PORT="${MAIN_PORT:-8000}"
# 5090 32GB のうち buzz サーバ(~3GB)・KVキャッシュ分を残す。
GPU_FRAC="${GPU_FRAC:-0.78}"
MAXLEN="${MAX_MODEL_LEN:-2048}"

# flashinfer サンプラーは runtime イメージで nvcc 要求 → JIT 失敗するため無効化。
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[main-serve] serving ${MODEL} on :${PORT} (gpu_frac=${GPU_FRAC}, max_len=${MAXLEN})"
exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${GPU_FRAC}" \
  --enable-prefix-caching \
  --stop-token-ids 1 106 \
  --served-model-name quiz-main
