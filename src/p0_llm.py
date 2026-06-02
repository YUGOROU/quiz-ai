# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
# ]
# ///
"""Phase 0: メインLLM（gpt-oss-120b 代替）の非同期ストリーミング・クライアント。

投機推論のため AsyncOpenAI を使い、in-flight リクエストを task.cancel() で
途中破棄できるようにする。回答とレイテンシ（TTFT/total）を返す。

主モデルは Phase 0 仕様どおり gpt-oss-120b:cerebras（HF Inference）。
HF_TOKEN が無ければ Crof.ai deepseek-v4-flash にフォールバック。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass


@dataclass
class LLMResult:
    answer: str
    reasoning: str
    ttft: float        # 最初のトークンまで（秒）
    total: float       # 生成完了まで（秒）
    raw: str


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_CLOSE_RE = re.compile(r"</think>")


def parse_answer(text: str) -> tuple[str, str]:
    """(answer, reasoning) を返す。<think>…</think>後ろを答えとみなす。"""
    reasoning = ""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m:
        reasoning = m.group(1).strip()
    tail = text[m.end():] if m else _THINK_RE.sub("", text)
    lines = [l.strip() for l in tail.strip().splitlines() if l.strip()]
    answer = lines[-1] if lines else text.strip()
    # 「答え:」等の接頭辞を除去
    answer = re.sub(r"^(答え|回答|Answer)\s*[:：]\s*", "", answer).strip()
    return answer, reasoning


class MainLLM:
    """投機推論用の非同期メインLLM。`answer()` は asyncio.Task でラップしてキャンセル可。"""

    SYSTEM = (
        "あなたは競技早押しクイズの解答者です。問題文は途中までしか与えられません。"
        "与えられた手がかりから最も可能性の高い答えを1つ推測してください。"
        "出力形式は <think>1〜2文で簡潔な根拠</think>答え（固有名詞を1語）。"
    )

    def __init__(self, model: str | None = None, max_tokens: int = 256,
                 temperature: float = 0.0, reasoning_effort: str | None = "low"):
        from openai import AsyncOpenAI

        self.max_tokens = max_tokens
        self.temperature = temperature
        # gpt-oss は reasoning_effort を絞ると思考トークンを節約でき、
        # 限られた max_tokens 内で最終回答が出やすくなる（None で無効）。
        self.extra_body: dict = {}
        if reasoning_effort:
            self.extra_body["reasoning_effort"] = reasoning_effort
        hf = os.environ.get("HF_TOKEN")
        crof = os.environ.get("CROFAI_API_KEY")
        override = os.environ.get("P0_MAIN_MODEL")
        if hf:
            self.client = AsyncOpenAI(base_url="https://router.huggingface.co/v1", api_key=hf)
            self.model = model or override or "openai/gpt-oss-120b:cerebras"
        elif crof:
            self.client = AsyncOpenAI(
                base_url=os.environ.get("CROFAI_BASE_URL", "https://crof.ai/v1"), api_key=crof)
            self.model = model or override or "deepseek-v4-flash"
        else:
            raise RuntimeError("HF_TOKEN（推奨）または CROFAI_API_KEY を設定してください。")

    async def answer(self, prefix: str) -> LLMResult:
        """部分問題文 prefix に対して回答をストリーミング生成する。

        asyncio.CancelledError は呼び出し側に伝播（投機の打ち切り）。

        gpt-oss 等の reasoning モデルは思考を `delta.reasoning(_content)` の
        別チャンネルに流し、`delta.content` には最終回答のみを入れる。両方を
        個別に集約し、回答は content（無ければ reasoning 末尾）から取り出す。
        """
        t0 = time.perf_counter()
        ttft: float | None = None
        content_chunks: list[str] = []
        reason_chunks: list[str] = []
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": f"問題文（途中まで）:\n{prefix}"},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
            extra_body=self.extra_body,
        )
        try:
            async for ev in stream:
                if not ev.choices:
                    continue
                delta = ev.choices[0].delta
                # 思考チャンネル（reasoning / reasoning_content）を別途集約
                rc = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                if rc:
                    reason_chunks.append(rc)
                txt = delta.content or ""
                if txt and ttft is None:
                    ttft = time.perf_counter() - t0
                content_chunks.append(txt)
        finally:
            # キャンセル時もHTTPストリームを確実に閉じる
            await stream.close()
        content = "".join(content_chunks)
        reason_field = "".join(reason_chunks)
        raw = content
        ans, reasoning = parse_answer(content)
        # content が空（max_tokens 切れ等で最終回答が出ない）の場合は
        # 思考チャンネル末尾を回答候補にフォールバックする
        if not ans and reason_field:
            ans, _ = parse_answer(reason_field)
            raw = f"[reasoning-only]{reason_field}"
        if not reasoning and reason_field:
            reasoning = reason_field.strip()
        return LLMResult(ans, reasoning, ttft or 0.0, time.perf_counter() - t0, raw)
