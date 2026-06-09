"""space/round_builder.build_round を実モデルで検証（Modal H100）。

JP デモ問題（questions_ja.json）に対し buzz位置・回答・思考・正誤を実生成し、
フロントエンドに渡る QUIZ_ROUND がデモとして成立するか（回答が正しいか・buzzFrac が
妥当か・aiThink が出るか）を確認する。Space に push する前のデモ品質チェック。

  uv run --with modal modal run space/validate_round_modal.py --theta 0.6
"""
import os

import modal

app = modal.App("quiz-round-validate")

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
def validate(lang: str, theta: float, load_4bit: bool):
    import json
    import sys
    import time

    sys.path.insert(0, "/opt/quizsrc")
    sys.path.insert(0, "/opt/space")
    import qutils
    import round_builder as rb

    data = json.loads(open(f"/opt/space/questions_{lang}.json", encoding="utf-8").read())
    qs = [{"id": q.get("id"), "full": q["full"], "truth": q["truth"],
           "category": q.get("category", ""), "pattern": q.get("pattern", "")}
          for q in data["questions"]]

    t0 = time.perf_counter()
    m = rb.load_models(token=os.environ.get("HF_TOKEN"), device="cuda", load_4bit=load_4bit)
    t_load = time.perf_counter() - t0
    print(f"[validate] load={t_load:.1f}s  questions={len(qs)}  θ={theta}")

    t1 = time.perf_counter()
    rnd = rb.build_round(qs, m, qutils=qutils, match=data.get("match", "Demo"),
                         theta=theta,
                         progress=lambda i, n, qid: print(f"  …Q{qid} ({i+1}/{n})"))
    t_build = time.perf_counter() - t1

    ok = 0
    print(f"\n[validate] === QUIZ_ROUND（{rnd['match']}）===")
    for q in rnd["questions"]:
        ok += q["correct"]
        mark = "✅" if q["correct"] else "❌"
        L = len(q["full"])
        bc = int(q["buzzFrac"] * L)
        print(f"{mark} Q{q['id']} buzz@{bc}/{L}({q['buzzFrac']:.0%}) "
              f"truth={q['truth']!r} answer={q['answer']!r} cross={q['aiCrossed']}")
        for s in q["aiThink"]:
            print(f"      think[{s['frac']:.2f}] {s['text']}")
    N = len(rnd["questions"]) or 1
    print(f"\n[validate] 正解 {ok}/{N}={ok/N:.0%}  build={t_build:.1f}s "
          f"（{t_build/N:.1f}s/問）  1マッチ生成≈{t_load + t_build:.0f}s")
    return rnd


@app.local_entrypoint()
def main(lang: str = "ja", theta: float = 0.6, gpu: str = "", load_4bit: bool = False):
    fn = validate.with_options(gpu=gpu) if gpu else validate
    rnd = fn.remote(lang=lang, theta=theta, load_4bit=load_4bit)
    print("\nquestions:", len(rnd.get("questions", [])))
