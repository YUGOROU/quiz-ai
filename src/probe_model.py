# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
# ]
# ///
"""Step1 アノテモデル候補の品質プローブ（Crof.ai）。

目的（qwen3.5-9b 採用判断の材料を一発で出す）:
  A) 思考オフが効くか      — extra_body 候補を総当たりし reasoning_tokens が 0 になる指定を特定
  B) usage の内訳          — prompt / completion / reasoning トークンを実測
  C) 早押しクイズ正解率     — 全文 / 65%prefix での正答を内蔵サンプルで集計
  D) 単価込みの実コスト感   — usage × Crof価格で 1コールあたり概算

使い方:
  export CROFAI_API_KEY=...
  export CROFAI_BASE_URL=https://crof.ai/v1   # 省略可
  uv run probe_model.py --model qwen3.5-9b --compare deepseek-v4-flash

  # Drive のアノテ済みデータで実問題を見たい場合（buzz_char と突き合わせ）
  uv run probe_model.py --model qwen3.5-9b --in /content/drive/MyDrive/quiz-ai/annotated_questions.jsonl --n 10
"""
from __future__ import annotations

import argparse
import os
import time

from openai import OpenAI

import qutils as U

# Crof.ai 価格（$/1M, 2026-05 時点。/v1/models より）
PRICES = {
    "qwen3.5-9b": (0.04, 0.15),
    "glm-4.7-flash": (0.04, 0.30),
    "deepseek-v4-flash": (0.12, 0.21),
    "deepseek-v3.2": (0.28, 0.38),
    "gemma-4-31b-it": (0.10, 0.30),
    "greg-1-mini": (0.07, 0.15),
}

# 思考オフ候補（Crof docs: reasoning_effort="none" で完全停止。
# SDK型検証を避けるため extra_body 経由で body に直接載せる）
THINK_OFF_CANDIDATES = [
    ("default(指定なし)", {}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
]

# 内蔵サンプル（早押し風・難易度ばらけ。最後はパラレル構文）
SAMPLES: list[tuple[str, list[str]]] = [
    ("日本でもっとも高い山で、標高3776メートル、古くから霊峰として信仰の対象となってきた山は何でしょう？", ["富士山"]),
    ("『吾輩は猫である』『坊っちゃん』などの作品で知られる、明治を代表する文豪は誰でしょう？", ["夏目漱石"]),
    ("アメリカ合衆国の初代大統領を務め、現在の1ドル紙幣に肖像が描かれている人物は誰でしょう？", ["ワシントン", "ジョージ・ワシントン"]),
    ("元素記号Fe、原子番号26で表される、人類が古くから道具や建材に利用してきた金属は何でしょう？", ["鉄"]),
    ("1602年にオランダで設立された、世界初の株式会社とも言われる貿易会社は何でしょう？", ["東インド会社", "オランダ東インド会社"]),
    ("太陽系の惑星のうち、もっとも内側を公転している、最も小さな惑星は何でしょう？", ["水星"]),
    ("シェイクスピアのいわゆる四大悲劇のうち、デンマークの王子を主人公とする作品は何でしょう？", ["ハムレット"]),
    ("日本国憲法が施行されたのは1947年の5月3日ですが、公布されたのは1946年の何月何日でしょう？", ["11月3日"]),
]

PROMPT_TMPL = (
    "以下は早押しクイズの問題文の途中（{n}文字目まで）です。\n"
    "問題: {prefix}\n"
    "答えは1語で。不明な場合は「不明」と答えてください。\nAnswer:"
)


def make_client() -> OpenAI:
    key = os.environ.get("CROFAI_API_KEY")
    if not key:
        raise SystemExit("CROFAI_API_KEY を設定してください。")
    base = os.environ.get("CROFAI_BASE_URL", "https://crof.ai/v1")
    return OpenAI(base_url=base, api_key=key)


def usage_fields(u) -> tuple[int, int, int]:
    """(prompt, completion, reasoning) トークンを防御的に取り出す。"""
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    rt = 0
    details = getattr(u, "completion_tokens_details", None)
    if details is not None:
        rt = getattr(details, "reasoning_tokens", 0) or 0
        if isinstance(details, dict):
            rt = details.get("reasoning_tokens", 0) or 0
    return pt, ct, rt


def ask(client, model, prompt, extra_body, max_tokens=64):
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body=extra_body or None,
    )
    dt = time.perf_counter() - t0
    msg = r.choices[0].message
    content = (msg.content or "").strip()
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
    return content, reasoning, r.usage, dt


def probe_think_off(client, model) -> dict:
    """思考を誘発しやすい問いで各 extra_body 候補を試し、reasoning_tokens を比較。"""
    print(f"\n── A) 思考オフ探索: {model} ──")
    prompt = "1から10までの整数を全て足すといくつ？答えの数字だけ出力して。"
    # 真のコスト信号は completion_tokens（CoTがcontent側に吐かれても課金されるため）。
    # 回答が空でなく completion が最小の指定を採用する。
    best = None  # (name, eb, completion_tokens)
    for name, eb in THINK_OFF_CANDIDATES:
        try:
            content, reasoning, u, dt = ask(client, model, prompt, eb, max_tokens=256)
            pt, ct, rt = usage_fields(u)
            off = ct <= 30 and content  # 答えだけ＝思考が実質オフ
            flag = "✅思考オフ" if off else "🟡思考あり"
            print(f"  [{name:24}] completion_tok={ct:4} reasoning_tok={rt:3} "
                  f"reasoning_field={'有' if reasoning else '無'} {flag}  ans={content[:20]!r}")
            if content and (best is None or ct < best[2]):
                best = (name, eb, ct)
        except Exception as e:  # noqa: BLE001
            print(f"  [{name:24}] エラー: {e}")
    if best:
        print(f"  → completion最小の採用候補: {best[0]}（completion={best[2]}tok）")
    else:
        print("  → 有効な回答を返す指定が無し")
    return dict(best=(best[0], best[1]) if best else None)


def probe_quality(client, model, extra_body, samples, ratios=(1.0, 0.65)) -> None:
    print(f"\n── C) 早押し正解率: {model}  extra_body={extra_body or '{}'} ──")
    pin, pout = PRICES.get(model, (0.0, 0.0))
    tally = {r: [0, 0] for r in ratios}  # ratio -> [correct, total]
    sum_pt = sum_ct = sum_rt = 0
    n_calls = 0
    for q, golds in samples:
        L = len(q)
        line = f"  gold={golds}"
        for r in ratios:
            n = max(1, int(L * r))
            prefix = q[:n]
            prompt = PROMPT_TMPL.format(n=n, prefix=prefix)
            content, reasoning, u, dt = ask(client, model, prompt, extra_body)
            pt, ct, rt = usage_fields(u)
            sum_pt += pt; sum_ct += ct; sum_rt += rt; n_calls += 1
            ok = U.is_correct(content, golds)
            tally[r][1] += 1
            tally[r][0] += int(ok)
            mark = "✅" if ok else "❌"
            line += f"\n    {int(r*100):3}% {mark} ans={content[:24]!r} (p{pt}/c{ct}/r{rt}, {dt:.2f}s)"
        print(line)
    print("  --- 集計 ---")
    for r in ratios:
        c, t = tally[r]
        print(f"  {int(r*100)}%入力 正解率: {c}/{t} = {c/t:.0%}")
    if n_calls:
        cost = (sum_pt * pin + (sum_ct) * pout) / 1e6
        print(f"  usage計: prompt={sum_pt} completion={sum_ct}(内reasoning={sum_rt}) "
              f"/ {n_calls}コール / 実コスト≈${cost:.4f}（${cost/n_calls*1000:.3f}/1kコール）")


def load_real(path: str, n: int) -> list[tuple[str, list[str]]]:
    out = []
    for rec in U.read_jsonl(path):
        if rec.get("is_valid") and rec.get("question") and rec.get("answers"):
            out.append((rec["question"], rec["answers"]))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--compare", default="", help="並べて比較する2つ目のモデル（例: deepseek-v4-flash）")
    ap.add_argument("--in", dest="infile", default="", help="実問題を使う場合のannotated_questions.jsonlパス")
    ap.add_argument("--n", type=int, default=8, help="--in 使用時のサンプル問数")
    args = ap.parse_args()

    client = make_client()
    samples = load_real(args.infile, args.n) if args.infile else SAMPLES
    print(f"[probe] サンプル {len(samples)}問 / source={'実データ' if args.infile else '内蔵'}")

    models = [args.model] + ([args.compare] if args.compare else [])
    for m in models:
        info = probe_think_off(client, m)
        eb = info["best"][1] if info["best"] else {}
        probe_quality(client, m, eb, samples)


if __name__ == "__main__":
    main()
