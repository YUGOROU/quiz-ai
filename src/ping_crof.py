# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
# ]
# ///
"""Crof.ai 接続テスト — 疎通・レイテンシ・空応答/思考課金の生確認。

qutils.chat はリトライ6回＋例外握りつぶしのため、Crof崩壊の症状（例外・空応答）が
隠れる。本スクリプトは生の create を1発ずつ叩き、失敗をそのまま表示する。

使い方（キーは自分で export。スクリプトはキー値に触れない）:
    cd /Users/yugoito/quiz-ai
    # ! export CROFAI_API_KEY=...    ← 各自のシェルで
    uv run src/ping_crof.py
Novita 経路で試す場合:
    unset CROFAI_API_KEY
    export HF_TOKEN=$(~/.local/bin/hf auth token) \
           SYNTH_MODEL="deepseek-ai/DeepSeek-V4-Flash:novita" \
           SYNTH_EXTRA_BODY='{"enable_thinking": false}'
    uv run src/ping_crof.py
"""
from __future__ import annotations

import json
import os
import time

from qutils import get_client

# 全文なら容易に答えられる平易な問題（空応答=serving異常の判定用）
QS = [
    "日本で一番高い山は？答えを1語で。",
    "『走れメロス』の作者は？答えを1語で。",
    "水の化学式は？答えを1語で。",
    "1600年に起きた天下分け目の戦いは？答えを1語で。",
    "太陽系で最も大きい惑星は？答えを1語で。",
]


def _udetail(u, group: str, field: str) -> int:
    d = getattr(u, group, None) if u else None
    return int(getattr(d, field, 0) or 0) if d else 0


def main() -> None:
    client, model = get_client()
    print(f"[client] model={model} base={getattr(client, 'base_url', '?')}")

    # Step1 と同条件：隠れ思考は止める（reasoning_effort + Novita向け enable_thinking）
    extra: dict = {"reasoning_effort": "none"}
    env_extra = os.environ.get("SYNTH_EXTRA_BODY")
    if env_extra:
        extra.update(json.loads(env_extra))
    print(f"[extra_body] {extra}\n")

    ok = 0
    lat: list[float] = []
    for i, q in enumerate(QS, 1):
        t = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q}],
                max_tokens=32,
                temperature=0.0,
                extra_body=extra or None,
            )
            dt = time.time() - t
            lat.append(dt)
            content = (r.choices[0].message.content or "").strip()
            u = getattr(r, "usage", None)
            comp = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
            reason = _udetail(u, "completion_tokens_details", "reasoning_tokens")
            cached = _udetail(u, "prompt_tokens_details", "cached_tokens")
            flag = "  <<空応答!>>" if not content else ""
            print(f"[{i}] {dt:5.2f}s ok  comp={comp} reason={reason} cached={cached}"
                  f"  -> {content!r}{flag}")
            if content:
                ok += 1
        except Exception as e:  # noqa: BLE001 — 障害の型をそのまま見たい
            print(f"[{i}] {time.time() - t:5.2f}s ERR  {type(e).__name__}: {e}")

    print(f"\n[result] 成功(非空) {ok}/{len(QS)}", end="")
    if lat:
        print(f"  / レイテンシ avg={sum(lat) / len(lat):.2f}s "
              f"min={min(lat):.2f}s max={max(lat):.2f}s")
    else:
        print()

    if ok == 0:
        print("[判定] 全件 例外 or 空応答 → Crof.ai 依然障害の疑い。Novita避難を継続。")
    elif ok < len(QS):
        print("[判定] 一部失敗＝不安定。本番投入は様子見、または Novita 継続が安全。")
    else:
        print("[判定] 全件正常応答。復旧の可能性。本番前にパイロットで歩留まりを再確認。")


if __name__ == "__main__":
    main()
