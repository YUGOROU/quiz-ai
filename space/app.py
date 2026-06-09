"""早押しクイズAI — HF Space（ZeroGPU + gradio.Server でカスタム frontend をフルページ配信）。

構成（docs/HANDOFF.md「🎯ハッカソン」「統合契約」「HF Space デプロイの罠」）:
  - ≤32B（gemma-4-26B-A4B + buzz LFM2.5-1.2B ≈ 27.2B）・日本特化・Gradio+HF Space・ZeroGPU。
  - **`gradio.Server`（FastAPI サブクラス・gradio 6）** を使う。標準 FastAPI ルート（`@app.get("/")`
    で React big-screen を配信・`@app.post("/api/round")` で実モデル生成）が使え、かつ
    **`app.launch()` が gradio launch 経路＝ZeroGPU の @spaces.GPU 検出を発火**させる
    （生 FastAPI+uvicorn は ZeroGPU 不可だったが Server なら両立。HANDOFF デプロイの罠参照）。
  - Claude Design の big-screen React を `static/` から `/` でフルページ配信。frontend は
    `POST /api/round`（実モデルで QUIZ_ROUND を 1 GPU 窓で precompute）を fetch して再生。

実装メモ:
  - モデルは ZeroGPU 推奨どおり module レベルで cuda 配置（startup CUDA エミュレーション）。
    既定 4bit（gemma ~15GB が `large` 48GB に収まり quota 1×）。bf16 は QUIZ_GPU_SIZE=xlarge。
  - gemma stop <turn|>(106)・thinking 必須・採点 qutils.is_correct。
  - QUIZ_MOCK=1 で GPU 無し UI 確認。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
for cand in (HERE / "src", HERE.parent / "src"):
    if cand.exists():
        sys.path.insert(0, str(cand))
        break
sys.path.insert(0, str(HERE))

import gradio as gr                 # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import qutils                       # noqa: E402
import round_builder as rb          # noqa: E402

import random  # noqa: E402

MAIN_REPO = os.environ.get("QUIZ_MAIN_REPO", rb.DEFAULT_MAIN_REPO)
BUZZ_REPO = os.environ.get("QUIZ_BUZZ_REPO", rb.DEFAULT_BUZZ_REPO)
LOAD_4BIT = os.environ.get("QUIZ_LOAD_4BIT", "1") == "1"  # 実績構成（4bitフラグ＝bnb経由・5-6/6 だった）
MOCK = os.environ.get("QUIZ_MOCK", "0") == "1"
DEFAULT_THETA = float(os.environ.get("QUIZ_THETA", "0.65"))  # 実績点（curatedで6/6だった運用点）
N_QUESTIONS = int(os.environ.get("QUIZ_N", "6"))             # 1マッチの問題数
LOAD_TTS = os.environ.get("QUIZ_TTS", "1") == "1"            # Irodori 読み上げ（QUIZ_TTS=0で無効）
BATCH_MAIN = os.environ.get("QUIZ_BATCH_MAIN", "1") == "1"   # gemma 生成を全問1バッチ化（初回待ち短縮）
TTS_REF = str(HERE / "announcer_ref.wav")                    # 参照音声（VoiceDesign アナウンサー声）
TTS_STEPS = int(os.environ.get("QUIZ_TTS_STEPS", "64"))      # num_steps=64（ユーザー確認の音質点）
STATIC = HERE / "static"

# 得意ジャンル候補（label_genres.py のタクソノミーと一致・順序固定）。
GENRES = [
    "文学・言葉", "歴史", "地理", "政治・経済", "社会・時事",
    "理科・科学", "数学", "医学・人体", "スポーツ", "野球",
    "音楽", "美術・建築", "映画・演劇", "アニメ・漫画・ゲーム", "アイドル・芸能",
    "テレビ・お笑い", "グルメ・料理", "生活・雑学", "動物・植物", "国際・宗教・神話",
]

# 問題プール。デモは curated（AIが確実に解ける有名定番・オリジナル）を優先。
# 無ければ hpprc/quiz-works の1500問プール（ランダム・多様だが難度高）にフォールバック。
_POOL: list[dict] = []
_curated_path = HERE / "questions_curated_ja.json"
_pool_path = _curated_path if _curated_path.exists() else (HERE / "questions_pool_ja.json")
if _pool_path.exists():
    _POOL = json.loads(_pool_path.read_text(encoding="utf-8")).get("questions", [])
    _labeled = sum(1 for q in _POOL if q.get("genre"))
    print(f"[app] question pool loaded: {len(_POOL)} (quiz-works), genre-labeled: {_labeled}")

# ZeroGPU の spaces。ローカル（未インストール）では恒等デコレータでスタブ。
try:
    import spaces
except Exception:  # noqa: BLE001
    class _SpacesStub:
        def GPU(self, *a, **k):
            if a and callable(a[0]):
                return a[0]
            return lambda fn: fn
    spaces = _SpacesStub()  # type: ignore

# モデル（ZeroGPU 推奨: module レベルで cuda 配置）。MOCK 時はロードしない。
_MODELS = None
if not MOCK:
    try:
        _MODELS = rb.load_models(MAIN_REPO, BUZZ_REPO, token=os.environ.get("HF_TOKEN"),
                                 device="cuda", load_4bit=LOAD_4BIT, load_tts=LOAD_TTS)
        print(f"[app] models loaded at module level (4bit={LOAD_4BIT}, tts={LOAD_TTS})")
    except Exception as e:  # noqa: BLE001 — startup で落とさず生成時に再試行
        print(f"[app] ⚠ module-level load 失敗（生成時に再試行）: {type(e).__name__}: {e}")


def _select_questions(n: int, genre: str | None = None) -> tuple[list[dict], str]:
    """1マッチぶんの問題を選ぶ。プールがあればランダム n 問、無ければ固定 JSON 先頭 n 問。
    genre 指定時はそのジャンルから優先的に出題（不足分は全体から補完）。
    返り値 (questions[{id,full,truth,genre?,...}], match名)。
    問題は日本語固定（日本特化が差別化軸）。UI 言語のみ EN/JA 切替する。"""
    if _POOL:
        pool = _POOL
        if genre and genre not in ("", "all", "おまかせ"):
            picked = [q for q in _POOL if q.get("genre") == genre]
            others = [q for q in _POOL if q.get("genre") != genre]
            if len(picked) >= n:
                return random.sample(picked, n), f"得意ジャンル: {genre}"
            # ジャンルの問題が n に満たなければ全部出し、残りを他から補う。
            fill = random.sample(others, min(n - len(picked), len(others)))
            qs = picked + fill
            random.shuffle(qs)
            return qs, f"得意ジャンル: {genre}"
        return random.sample(pool, min(n, len(pool))), "早押しクイズAI デモマッチ"
    data = json.loads((HERE / "questions_ja.json").read_text(encoding="utf-8"))
    return data["questions"][:n], data.get("match", "Live Match")


@spaces.GPU(duration=120)
def _gpu_build(questions: list[dict], match: str, theta: float) -> dict:
    global _MODELS
    if _MODELS is None:
        _MODELS = rb.load_models(MAIN_REPO, BUZZ_REPO, token=os.environ.get("HF_TOKEN"),
                                 device="cuda", load_4bit=LOAD_4BIT, load_tts=LOAD_TTS)
    return rb.build_round(questions, _MODELS, qutils=qutils, match=match, theta=theta,
                          batch_main=BATCH_MAIN,
                          tts_ref=TTS_REF if LOAD_TTS else None, tts_steps=TTS_STEPS)


def _mock_round(qs: list[dict], match: str) -> dict:
    out = [{"id": q.get("id"), "category": q.get("category", ""),
            "pattern": q.get("pattern", ""), "genre": q.get("genre", ""),
            "full": q["full"], "truth": q["truth"],
            "buzzer": "ai", "buzzFrac": 0.6, "answer": q["truth"], "correct": True,
            "aiCrossed": True,
            "aiThink": [{"frac": 0.3, "text": "（mock）手がかりを解析中…"},
                        {"frac": 0.55, "text": f"（mock）{q['truth']} と予測"}]}
           for q in qs]
    return {"match": match, "questions": out}


def build_round_api(genre: str | None = None, theta: float | None = None) -> dict:
    qs_sel, match = _select_questions(N_QUESTIONS, genre)
    if MOCK:
        return _mock_round(qs_sel, match)
    qs = [{"id": q.get("id"), "full": q["full"], "truth": q["truth"],
           "category": q.get("category", ""), "pattern": q.get("pattern", ""),
           "genre": q.get("genre", "")}
          for q in qs_sel]
    return _gpu_build(qs, match, DEFAULT_THETA if theta is None else theta)


# ============================================================
# gradio.Server（FastAPI サブクラス）— ZeroGPU 互換のまま custom frontend を配信
# ============================================================
app = gr.Server(title="quiz-buzzer-ai")


@app.get("/api/health")
def health():
    return {"status": "ok", "mock": MOCK, "main": MAIN_REPO, "buzz": BUZZ_REPO,
            "models_ready": _MODELS is not None}


@app.get("/api/genres")
def api_genres():
    """得意ジャンル候補。プールに実在しラベル済みのジャンルのみ（件数付き）を返す。"""
    from collections import Counter
    dist = Counter(q.get("genre", "") for q in _POOL if q.get("genre"))
    avail = [{"genre": g, "count": dist[g]} for g in GENRES if dist.get(g, 0) > 0]
    return {"genres": avail, "min_for_match": N_QUESTIONS}


# ★ ZeroGPU は **並行する @spaces.GPU 呼び出しごとに別GPUを確保**する（マルチGPU仕様）。
# 複数タブ/端末/観客が生成中（~36s）に同時に Start を押すと GPU を2個以上つかみ quota が
# 倍速で溶ける。ここで single-flight 直列化し、常に1個だけ確保されるようにする
# （2人目以降は GPU を要求する前にロック待ち＝順番に1個ずつ処理）。
_BUILD_LOCK = threading.Lock()


@app.post("/api/round")
def api_round(payload: dict | None = None):
    payload = payload or {}
    try:
        with _BUILD_LOCK:                       # 直列化（同時に2個確保しない）
            t0 = time.time()                    # 実ビルド秒（キュー待ちは含めない）
            result = build_round_api(payload.get("genre"), payload.get("theta"))
            result["_build_seconds"] = round(time.time() - t0, 1)  # frontend の ETA 自己較正用
        return JSONResponse(result)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)[:400]}, status_code=500)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    # HF Space は `python app.py` を実行 → app.launch() がブロッキングで 7860 を配信し、
    # gradio launch 経路で ZeroGPU の @spaces.GPU 検出が発火する。
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
