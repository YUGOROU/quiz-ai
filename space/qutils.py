"""早押しクイズAI — 解答判定ユーティリティ（HF Space 用スリム版）。

Space が必要とするのは `is_correct`（と内部の正規化）だけ。本体 `src/qutils.py` の
LLMクライアント・UsageMeter・JSONL I/O・コーパスDL 等（データ合成パイプライン用）は
Space には不要なので含めない。判定ロジックは `src/qutils.py` と同一に保つこと。

標準ライブラリのみ依存。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# ── 正規化・正誤判定（src/qutils.py と同一ロジック） ──────────────────

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


# ── loose（早押し読み上げ判定相当の緩和） ──────────────────────────
_LOOSE_DROP_RE = re.compile(r"[ー・･\s]")
_SMALL_TO_FULL = str.maketrans("ァィゥェォャュョッヮ", "アイウエオヤユヨツワ")


def _loose_form(text: str) -> str:
    """normalize の上に、長音「ー」・中黒・空白を除去し小書きカナを大書きに畳む。"""
    t = normalize(text)
    t = _LOOSE_DROP_RE.sub("", t)
    return t.translate(_SMALL_TO_FULL)


def _levenshtein(a: str, b: str) -> int:
    """編集距離（loose の僅差許容に使う・外部依存なしの2行DP）。"""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_correct(pred: str, golds: Iterable[str], subset_min_len: int = 2, *,
               loose: bool = False, max_edits: int = 1,
               loose_min_len: int = 4) -> bool:
    """予測が正解集合のいずれかに一致すれば True（正規化後）。

    完全一致・gold⊂pred に加え、len(pred)>=subset_min_len のときだけ pred⊂gold も許容。
    loose=True で長音/中黒/空白/小書きカナを吸収し、双方 loose_min_len 以上のとき編集距離
    max_edits 以下も救済（距離2の翻字違いは通さない）。src/qutils.py と同一。
    """
    norm = _loose_form if loose else normalize
    p = norm(pred)
    if not p:
        return False
    for g in golds:
        ng = norm(g)
        if not ng:
            continue
        if ng == p or ng in p:
            return True
        if len(p) >= subset_min_len and p in ng:
            return True
        if (loose and max_edits > 0 and min(len(p), len(ng)) >= loose_min_len
                and _levenshtein(p, ng) <= max_edits):
            return True
    return False
