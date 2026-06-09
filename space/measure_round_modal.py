"""build_round の GPU 使用量・速度を計測（Modal H100）。

ZeroGPU（RTX Pro 6000 Blackwell 96GB）でバッチ生成を導入する余地を測る:
  - sequential（現状: gemma 6問逐次）vs batched（全問1バッチ）の peak VRAM / wall time
  - rebound precompute（AI 全文回答を追加生成）込みの VRAM
  - 4bit ロード時の常駐 VRAM と、生成時の追加（活性化）VRAM

  uv run --with modal modal run space/measure_round_modal.py --n 6 --load-4bit
"""
import os

import modal

app = modal.App("quiz-round-measure")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch", "transformers", "accelerate", "huggingface_hub", "hf_transfer",
        "sentencepiece", "protobuf", "einops", "pillow", "tiktoken", "blobfile",
        "triton", "bitsandbytes",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
    .add_local_dir(_HERE, "/opt/space", copy=True,
                   ignore=["__pycache__", "*.pyc", "static"])
    .add_local_dir(_SRC, "/opt/quizsrc", copy=True, ignore=["__pycache__", "*.pyc"])
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(image=image, gpu="H100", cpu=4.0, timeout=60 * 30,
              volumes={"/hf": hf_cache}, secrets=secrets)
def measure(n: int, theta: float, load_4bit: bool):
    import json
    import sys
    import time

    import torch

    sys.path.insert(0, "/opt/quizsrc")
    sys.path.insert(0, "/opt/space")
    import qutils
    import round_builder as rb

    def gib(x):
        return x / (1024 ** 3)

    def vram():
        return gib(torch.cuda.memory_allocated()), gib(torch.cuda.max_memory_allocated())

    total = gib(torch.cuda.get_device_properties(0).total_memory)
    print(f"[measure] GPU total={total:.1f} GiB  name={torch.cuda.get_device_name(0)}")

    # 問題プール優先（無ければ固定JSON）。n 問抽出。
    pool_path = "/opt/space/questions_pool_ja.json"
    src = pool_path if os.path.exists(pool_path) else "/opt/space/questions_ja.json"
    data = json.loads(open(src, encoding="utf-8").read())
    qs = [{"id": q.get("id"), "full": q["full"], "truth": q["truth"],
           "genre": q.get("genre", "")} for q in data["questions"][:n]]
    print(f"[measure] questions={len(qs)} from {os.path.basename(src)}")

    t0 = time.perf_counter()
    m = rb.load_models(token=os.environ.get("HF_TOKEN"), device="cuda", load_4bit=load_4bit)
    t_load = time.perf_counter() - t0
    resident, _ = vram()
    print(f"[measure] load={t_load:.1f}s  resident VRAM(モデル常駐)={resident:.1f} GiB (4bit={load_4bit})")

    def run(tag, **kw):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        rnd = rb.build_round(qs, m, qutils=qutils, theta=theta, **kw)
        dt = time.perf_counter() - t
        cur, peak = vram()
        ok = sum(q["correct"] for q in rnd["questions"])
        N = len(rnd["questions"]) or 1
        print(f"[measure] {tag:32} time={dt:5.1f}s ({dt/N:4.1f}s/問)  peak VRAM={peak:5.1f} GiB "
              f"(+{peak-resident:4.1f} 活性化)  正解={ok}/{N}")
        return rnd, dt, peak

    print("\n[measure] ── 比較 ──")
    seq, t_seq, _ = run("sequential (現状・rebound無)", batch_main=False, rebound=False)
    bat, t_bat, peak_bat = run("batched (rebound無)", batch_main=True, rebound=False)
    batr, t_batr, peak_batr = run("batched + rebound (全文回答も)", batch_main=True, rebound=True)

    # 回答一致チェック（greedy なので sequential と batched は一致するはず）
    same = all(a["answer"] == b["answer"] for a, b in zip(seq["questions"], bat["questions"]))
    print(f"\n[measure] sequential と batched の回答一致: {same}")
    print(f"[measure] 速度: seq {t_seq:.1f}s → batched {t_bat:.1f}s "
          f"（{t_seq/max(t_bat,0.01):.2f}x）  rebound込み {t_batr:.1f}s")
    print(f"[measure] VRAM余裕: batched+rebound peak {peak_batr:.1f} / {total:.0f} GiB "
          f"= {100*peak_batr/total:.0f}% 使用。残り {total-peak_batr:.0f} GiB")
    # rebound の AI 全文回答サンプル
    print("\n[measure] rebound 用 aiFullAnswer サンプル:")
    for q in batr["questions"][:n]:
        print(f"  Q{q['id']} buzz回答={q['answer']!r}({q['correct']}) "
              f"全文回答={q.get('aiFullAnswer')!r}({q.get('aiFullCorrect')}) truth={q['truth']!r}")
    return {"t_seq": t_seq, "t_bat": t_bat, "t_batr": t_batr,
            "peak_batr": peak_batr, "total": total}


@app.local_entrypoint()
def main(n: int = 6, theta: float = 0.6, load_4bit: bool = True, gpu: str = ""):
    fn = measure.with_options(gpu=gpu) if gpu else measure
    r = fn.remote(n=n, theta=theta, load_4bit=load_4bit)
    print("\n[result]", r)
