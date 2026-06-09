---
title: Quiz Buzzer AI
emoji: ⚡
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: true
license: other
short_description: Head-to-head Japanese competitive buzz-quiz against an AI
---

# Quiz Buzzer AI ⚡ — 早押しクイズAI

A head-to-head **competitive buzz-quiz** (早押しクイズ) where a human plays against an AI under
equal conditions. Built for the **HF Build Small Hackathon** (total params ≤ 32B).

> 早押しクイズ (hayaoshi quiz) is a Japanese competitive-quiz format: the question is read aloud and
> players race to **buzz in as early as they dare** — buzzing too early on the lead-in clue is a costly
> false start. This project is **specialized for Japanese**; questions are in 日本語.

## Links
- 💻 **Source code (GitHub):** https://github.com/YUGOROU/quiz-ai
- 🧠 **Answering model:** https://huggingface.co/YUGOROU/quiz-main-gemma-merged
- 🔔 **Buzz-timing model:** https://huggingface.co/YUGOROU/quiz-buzz-reg-1.2bjp-merged

## How it works (two fine-tuned models, ≤ 32B total)

| Role | Model | Job |
|---|---|---|
| **Buzz timing** | `YUGOROU/quiz-buzz-reg-1.2bjp-merged` (LFM2.5-1.2B + regression head) | Reads the question char-by-char, emits a confidence; buzzes when `conf ≥ θ`. ~9 ms/char. |
| **Answering** | `YUGOROU/quiz-main-gemma-merged` (gemma-4-26B-A4B SFT) | From the partial question at buzz time, reasons (`<think>…</think>`) and answers. |

Both are **fine-tuned** on a quiz-grammar corpus derived from AI王 / JAQKET (corpus private; weights +
inference code only). Total ≈ 27.2B params.

## Architecture

- **ZeroGPU**: one `@spaces.GPU(duration=120)` window **precomputes a whole match** (buzz position,
  reasoning, answer, correctness for N questions). The frontend then plays it back smoothly, streaming
  the question as mock-STT at 22 chars/s while the human can buzz in live (**Space** / tap).
- **Frontend**: the spectator "Big Screen" UI (1920×1080) is a custom React frontend mounted on the
  Gradio app's FastAPI. It fetches the live match from `POST /api/round`.
- **Scoring**: correct = `1.0 + 0.5·(1 − buzzFrac)` (earlier buzz → bigger reward); wrong = `−1.5`.

## Endpoints
- `/` — Big Screen live UI (Start match → AI plays in real time)
- `/api/round` — generate a match with the real models (ZeroGPU)
- `/gradio` — minimal control panel (language, buzz θ)

## Env
- `QUIZ_MAIN_REPO` / `QUIZ_BUZZ_REPO` — model repos (defaults above)
- `QUIZ_THETA` — buzz threshold (default 0.6; higher = more cautious = later buzz = higher accuracy)
- `QUIZ_LOAD_4BIT=1` — load the main model in 4bit (lighter)
- `QUIZ_MOCK=1` — frontend dev without GPU (returns a dummy match)

## Attribution / license
Training data derived from AI王 (Project AIO) / JAQKET. Quiz questions © abc/EQIDEN実行委員会 /
株式会社キュービック / クイズ法人カプリティオ. Non-commercial research use only. No dataset
redistribution. The demo question pool is **not** committed to GitHub; regenerate it locally with
`build_aio_pool.py` (downloads AI王 `data/aio`, CC BY-SA 4.0) + genre labelling, then place the
resulting `questions_pool_ja.json` next to `app.py`.

`irodori_tts/` is vendored from [Aratako/Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
(reads the question aloud) and retains its upstream license. The announcer reference voice
(`announcer_ref.wav`) is likewise not committed — supply your own reference clip.
