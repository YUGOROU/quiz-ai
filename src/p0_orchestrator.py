# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
# ]
# ///
"""Phase 0: 早押しパイプラインの asyncio オーケストレータ。

設計リスクの検証対象:
  - 投機推論の打ち切り機構（no-buzz/前提ズレで in-flight をキャンセル）
  - buzz確定時のコミット
  - メインLLMの prefix 正解率とレイテンシ

char-source と buzz判定を差し替え可能にしてある:
  - Phase 0a: char-source = テキストを時間供給（sim_char_source）/ buzz = ルール65%
  - Phase 0b: char-source = parakeet STT出力 / buzz = LFM2.5-350M（同インターフェース）
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from p0_llm import LLMResult, MainLLM


@dataclass
class EpisodeConfig:
    buzz_ratio: float = 0.65      # ルールbuzz発火位置（問題文長に対する割合）
    spec_lead: int = 8            # buzzの何文字手前で投機を開始するか
    char_rate: float = 12.0       # 文字供給レート（chars/sec）。STT速度の模擬
    realtime: bool = True         # Trueで実時間シミュレート（投機の先行を計測可能）
    reset_at: int | None = None   # 指定位置で前提ズレを模擬し投機をキャンセル→再起動


@dataclass
class EpisodeResult:
    qid: str
    answer: str
    correct: bool
    buzz_pos: int
    question_length: int
    t_buzz: float                 # buzz確定までの経過秒
    t_answer: float               # 回答確定までの経過秒
    buzz_to_answer: float         # t_answer - t_buzz（負に近いほど投機が効いている）
    spec_done_before_buzz: bool   # buzz時点で投機が完了済みだったか
    ttft: float
    llm_total: float
    n_cancellations: int
    reasoning: str = ""
    error: str | None = None


async def sim_char_source(question: str, char_rate: float, realtime: bool) -> AsyncIterator[tuple[int, str]]:
    """テキストを1文字ずつ (position, prefix) で供給する（0a用）。"""
    for pos in range(1, len(question) + 1):
        if realtime:
            await asyncio.sleep(1.0 / char_rate)
        yield pos, question[:pos]


def rule_buzz(pos: int, prefix: str, buzz_pos: int) -> bool:
    """ルールベースbuzz: 指定位置以降でbuzz（0a用）。"""
    return pos >= buzz_pos


async def run_episode(
    llm: MainLLM,
    qid: str,
    question: str,
    golds: list[str],
    cfg: EpisodeConfig,
    is_correct_fn: Callable[[str, list[str]], bool],
) -> EpisodeResult:
    L = len(question)
    buzz_pos = max(1, int(L * cfg.buzz_ratio))
    spec_pos = max(1, buzz_pos - cfg.spec_lead)

    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0

    spec_task: asyncio.Task | None = None
    n_cancel = 0
    t_buzz = t_answer = 0.0
    spec_done_before_buzz = False
    result: LLMResult | None = None
    error: str | None = None

    try:
        async for pos, prefix in sim_char_source(question, cfg.char_rate, cfg.realtime):
            # ── 投機開始（buzz手前）
            if spec_task is None and pos >= spec_pos:
                spec_task = asyncio.create_task(llm.answer(prefix))

            # ── 前提ズレの模擬：in-flight をキャンセルし、現在末尾で再起動
            if cfg.reset_at is not None and pos == cfg.reset_at and spec_task is not None:
                spec_task.cancel()
                try:
                    await spec_task
                except asyncio.CancelledError:
                    pass
                n_cancel += 1
                spec_task = asyncio.create_task(llm.answer(prefix))

            # ── buzz確定 → コミット
            if rule_buzz(pos, prefix, buzz_pos):
                t_buzz = now()
                if spec_task is None:
                    spec_task = asyncio.create_task(llm.answer(prefix))
                spec_done_before_buzz = spec_task.done()
                result = await spec_task
                t_answer = now()
                break
    except Exception as e:  # noqa: BLE001
        error = str(e)
        if spec_task and not spec_task.done():
            spec_task.cancel()

    if result is None and error is None:
        error = "no result"

    ans = result.answer if result else ""
    return EpisodeResult(
        qid=qid, answer=ans,
        correct=bool(result) and is_correct_fn(ans, golds),
        buzz_pos=buzz_pos, question_length=L,
        t_buzz=round(t_buzz, 4), t_answer=round(t_answer, 4),
        buzz_to_answer=round(t_answer - t_buzz, 4),
        spec_done_before_buzz=spec_done_before_buzz,
        ttft=round(result.ttft, 4) if result else 0.0,
        llm_total=round(result.total, 4) if result else 0.0,
        n_cancellations=n_cancel,
        reasoning=result.reasoning if result else "",
        error=error,
    )
