# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "tqdm",
# ]
# ///
"""早押しクイズLM — コーパス前処理 共有ユーティリティ。

正規化・LLMクライアント・リトライ・qid分割・JSONL I/O を全工程で共有する。
標準ライブラリ + openai のみ依存（pandas/numpy 不使用）。

環境変数:
  CROFAI_API_KEY   本命: Crof.ai deepseek-v4-flash の APIキー（必須）
  CROFAI_BASE_URL  省略時 "https://crof.ai/v1"
  SYNTH_MODEL      省略時 "deepseek-v4-flash"
  HF_TOKEN         フォールバック: gpt-oss-120b:cerebras
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from typing import Iterable, Iterator

# ── LLMクライアント ────────────────────────────────────────────────

def get_client():
    """(client, model_name) を返す。Crof.ai 優先、無ければ HF にフォールバック。"""
    from openai import OpenAI

    key = os.environ.get("CROFAI_API_KEY")
    if key:
        base = os.environ.get("CROFAI_BASE_URL", "https://crof.ai/v1")
        model = os.environ.get("SYNTH_MODEL", "deepseek-v4-flash")
        return OpenAI(base_url=base, api_key=key), model

    hf = os.environ.get("HF_TOKEN")
    if hf:
        return OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf), "openai/gpt-oss-120b:cerebras"

    raise RuntimeError(
        "APIの認証情報がありません。CROFAI_API_KEY と CROFAI_BASE_URL "
        "（または HF_TOKEN）を環境変数に設定してください。"
    )


def chat(client, model, prompt: str, *, max_tokens: int = 64,
         temperature: float = 0.0, retries: int = 6) -> str | None:
    """1ターンのchat補完。429/一時エラーは exponential backoff で再試行。

    全試行失敗時は None を返す（呼び出し側でスキップ扱い）。
    """
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 — provider横断で広く捕捉
            if attempt == retries - 1:
                return None
            # 429やネットワーク揺らぎを想定。指数バックオフ + 軽いジッタ。
            time.sleep(delay + (os.urandom(1)[0] / 255.0))
            delay = min(delay * 2, 30.0)
    return None


# ── 正規化・正誤判定 ──────────────────────────────────────────────

_PAREN_RE = re.compile(r"[（(][^）)]*[）)]")
_HONORIFIC_RE = re.compile(r"(さん|くん|氏|博士|先生)$")


def normalize(text: str) -> str:
    """日本語表記ゆれの正規化（全工程で統一して使う）。"""
    text = unicodedata.normalize("NFKC", text)
    text = _PAREN_RE.sub("", text)           # 括弧内の読み仮名を除去
    text = _HONORIFIC_RE.sub("", text.strip())  # 末尾の敬称を除去
    return text.strip()


def is_correct(pred: str, golds: Iterable[str]) -> bool:
    """予測が正解集合のいずれかに一致（正規化後・完全一致 or 包含）すれば True。"""
    p = normalize(pred)
    if not p:
        return False
    for g in golds:
        ng = normalize(g)
        if ng and (ng == p or ng in p):
            return True
    return False


# ── qid 単位の決定論的 split ──────────────────────────────────────

def qid_split(qid: str, train: float = 0.85, val: float = 0.10) -> str:
    """qid のハッシュで train/val/test を決定論的に割り当てる（シード固定不要）。

    corpus-1/2/3 で同じ関数を使い、同一問題が分割をまたがないようにする。
    """
    h = int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16)
    r = (h % 10_000) / 10_000.0
    if r < train:
        return "train"
    if r < train + val:
        return "val"
    return "test"


# ── JSONL I/O ─────────────────────────────────────────────────────

def read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_qids(path: str, key: str = "qid") -> set[str]:
    """キャッシュから処理済みキーを読み込む（中断再開用）。"""
    done: set[str] = set()
    for rec in read_jsonl(path):
        if key in rec:
            done.add(rec[key])
    return done


class JsonlWriter:
    """スレッドセーフな append 書き込み（逐次フラッシュで中断耐性）。"""

    def __init__(self, path: str):
        import threading
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        self._f.close()


# ── データダウンロード（stdlibのみ） ──────────────────────────────

DATA_URLS = {
    "aio_02_train.jsonl": "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_train.jsonl",
    "aio_02_dev.jsonl":   "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_dev_v1.0.jsonl",
    "aio_04_dev.jsonl":   "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_04/aio_04_dev_v1.0.jsonl",
}


def ensure_raw(name: str, raw_dir: str = "corpus/raw") -> str:
    """raw/ に未取得ならダウンロードしてパスを返す。"""
    import urllib.request

    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = DATA_URLS[name]
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, path)
    print(f"[download] -> {path} ({os.path.getsize(path):,} bytes)")
    return path
