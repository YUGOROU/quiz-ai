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
        # HF Inference Providers 経由。SYNTH_MODEL で provider 込みのモデルIDを差し替え可能。
        # 例) Crof.ai 障害時の一時避難: SYNTH_MODEL="deepseek-ai/DeepSeek-V4-Flash:novita"
        # 既定はRL対戦相手にも使う gpt-oss-120b:cerebras。
        # 注意: Novita等に逃がす時は CROFAI_API_KEY を未設定にする（このブランチに入れるため）。
        model = os.environ.get("SYNTH_MODEL", "openai/gpt-oss-120b:cerebras")
        return OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf), model

    raise RuntimeError(
        "APIの認証情報がありません。CROFAI_API_KEY と CROFAI_BASE_URL "
        "（または HF_TOKEN）を環境変数に設定してください。"
    )


class UsageMeter:
    """スレッドセーフな usage 集計（prompt/completion/reasoning トークン・コール数）。

    Step1 の本番投入前に reasoning が課金されていないか実測するために使う。
    """

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt = 0
        self.cached = 0       # プロンプトキャッシュにヒットした入力トークン
        self.completion = 0
        self.reasoning = 0

    @staticmethod
    def _detail(obj, field: str) -> int:
        if obj is None:
            return 0
        if isinstance(obj, dict):
            return obj.get(field, 0) or 0
        return getattr(obj, field, 0) or 0

    def add(self, usage) -> None:
        if usage is None:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        rt = self._detail(getattr(usage, "completion_tokens_details", None), "reasoning_tokens")
        # キャッシュヒット数は prompt_tokens_details.cached_tokens に入る（OpenAI互換）
        cached = self._detail(getattr(usage, "prompt_tokens_details", None), "cached_tokens")
        with self._lock:
            self.calls += 1
            self.prompt += pt
            self.cached += cached
            self.completion += ct
            self.reasoning += rt

    def summary(self, price: tuple[float, ...] | None = None) -> str:
        """price=(in,out) または (in,cache,out) $/1M を渡すと概算コストも返す。"""
        hit = f"{self.cached / self.prompt:.0%}" if self.prompt else "-"
        msg = (f"calls={self.calls:,} prompt={self.prompt:,}（内cache={self.cached:,}/{hit}）"
               f" completion={self.completion:,}（内reasoning={self.reasoning:,}）")
        if price and self.calls:
            if len(price) >= 3:
                p_in, p_cache, p_out = price[0], price[1], price[2]
                uncached = max(self.prompt - self.cached, 0)
                cost = (uncached * p_in + self.cached * p_cache + self.completion * p_out) / 1e6
            else:
                cost = (self.prompt * price[0] + self.completion * price[1]) / 1e6
            msg += f" / 実コスト≈${cost:.4f}（${cost / self.calls * 1000:.3f}/1kコール）"
        return msg


def chat(client, model, prompt: str, *, max_tokens: int = 64,
         temperature: float = 0.0, retries: int = 6,
         reasoning_effort: str | None = None,
         meter: "UsageMeter | None" = None) -> str | None:
    """1ターンのchat補完。429/一時エラーは exponential backoff で再試行。

    reasoning_effort: "none"/"low"/... を指定すると extra_body で思考量を制御する。
        Step1（アノテ）は答え1語しか要らないので "none" で隠れ思考の課金を止める。
        SDK の enum 検証を避けるため named ではなく extra_body で body に直接載せる。
    meter: 渡すと usage を集計する（本番前のコスト実測用）。

    全試行失敗時は None を返す（呼び出し側でスキップ扱い）。
    """
    extra_body: dict = {}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    # provider依存の追加bodyを環境変数で注入（呼び出し側を触らず切替）。
    # 例) Novita の DeepSeek-V4-Flash で隠れ思考を切る:
    #     SYNTH_EXTRA_BODY='{"enable_thinking": false}'
    #   （Crofの reasoning_effort="none" はNovita router では無視されるため。top-levelで渡す）
    _env_extra = os.environ.get("SYNTH_EXTRA_BODY")
    if _env_extra:
        extra_body.update(json.loads(_env_extra))
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body or None,
            )
            if meter is not None:
                meter.add(getattr(resp, "usage", None))
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
# タイトル等の引用括弧・装飾クォートは表記ゆれの元なので除去（中身は残す）
_QUOTE_RE = re.compile(r"[「」『』【】〔〕《》〈〉“”‘’\"']")
_WAVE_RE = re.compile(r"[〜～~]")           # 波ダッシュの揺れ（〜/～/~）を吸収


def _to_katakana(text: str) -> str:
    """ひらがな→カタカナに畳んでかな表記ゆれを吸収（いんげん vs インゲン）。"""
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def normalize(text: str) -> str:
    """日本語表記ゆれの正規化（全工程で統一して使う）。"""
    text = unicodedata.normalize("NFKC", text)
    text = _PAREN_RE.sub("", text)           # 括弧内の読み仮名を除去
    text = _QUOTE_RE.sub("", text)           # 引用括弧・クォートを除去（中身は残す）
    text = _WAVE_RE.sub("", text)            # 波ダッシュの揺れを吸収
    text = _HONORIFIC_RE.sub("", text.strip())  # 末尾の敬称を除去
    return _to_katakana(text.strip())        # 最後にかな統一


def is_correct(pred: str, golds: Iterable[str], subset_min_len: int = 2) -> bool:
    """予測が正解集合のいずれかに一致すれば True（正規化後）。

    完全一致・gold⊂pred に加え、len(pred)>=subset_min_len のときだけ pred⊂gold も許容する。
    これで「26⊂26文字」「アガサ・クリスティ⊂…ー」を救済しつつ、"1"⊂"1600年" のような
    短すぎる予測の誤マッチは長さガードで阻止する。早まる buzz_char は共同GRPOで再収束する前提。
    """
    p = normalize(pred)
    if not p:
        return False
    for g in golds:
        ng = normalize(g)
        if not ng:
            continue
        if ng == p or ng in p:
            return True
        if len(p) >= subset_min_len and p in ng:
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
