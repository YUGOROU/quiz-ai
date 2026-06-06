"""gemma-4 が現行 vLLM でロード/推論できるか実機確認（デプロイ必須要件）。

eval は transformers+trust_remote_code で動いたが、**デプロイは vLLM 前提**
（docs/quiz-ai.md）。vLLM はリポジトリ同梱コードを実行せず**ネイティブ実装のarch
のみ**読むため、gemma4(`Gemma4ForConditionalGeneration`) が現行 vLLM に入っているかは別問題。
ここで LLM ロード＋1件生成まで通るかを確認する。

  uv run --with modal modal run train/check_gemma4_vllm.py --repo google/gemma-4-26B-A4B
"""
import os

import modal

app = modal.App("quiz-gemma4-vllm-check")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "deep_gemm", "huggingface_hub", "hf_transfer")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/hf",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # flashinfer JIT が nvcc 要求で落ちる回避
    })
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(image=image, gpu="H100", volumes={"/hf": hf_cache},
              secrets=secrets, timeout=60 * 40)
def check(repo: str):
    import traceback
    try:
        from vllm import LLM, SamplingParams
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return {"repo": repo, "vllm_import": False}
    import vllm
    print(f"[vllm] version={vllm.__version__}  loading {repo} ...")
    try:
        llm = LLM(model=repo, dtype="bfloat16", max_model_len=2048,
                  gpu_memory_utilization=0.90, enforce_eager=True,
                  trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[vllm] LOAD 失敗: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
        return {"repo": repo, "vllm": vllm.__version__, "loaded": False}
    out = llm.generate(
        ["問題: 日本一高い山は？\n答え:"],
        SamplingParams(temperature=0.0, max_tokens=16),
    )
    text = out[0].outputs[0].text
    print(f"[vllm] OK 生成: {text!r}")
    return {"repo": repo, "vllm": vllm.__version__, "loaded": True, "sample": text}


@app.local_entrypoint()
def main(repo: str = "google/gemma-4-26B-A4B", gpu: str = ""):
    fn = check.with_options(gpu=gpu) if gpu else check
    print("result:", fn.remote(repo=repo))
