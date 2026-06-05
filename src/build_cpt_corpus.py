# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets",
#   "transformers",
#   "tqdm",
# ]
# ///
"""CPT（継続事前学習）用コーパス整形。range3/wiki40b-ja → cpt_corpus/{train,val}.jsonl。

背景（docs/HANDOFF.md）: メインLLM(Qwen3.5-9B-Base)の全文正解率48%は9Bの知識ハード上限。
SFT/ハイパラ/elicitation では動かず、parametric知識を上げる CPT が唯一のレバーと決着。
コーパスは「幅広いが小さい」日本語 wiki（AI王正答のみ=過適応を避けるユーザー指示）。
本命 range3/wiki40b-ja は wiki40b（Google品質フィルタ版）の日本語抽出・CC-BY-SA。

このスクリプトがやること:
  1. range3/wiki40b-ja train を DL（HFキャッシュ利用）
  2. wiki40b 特有のマークアップ（_START_ARTICLE_ 等）を素のプレーンテキストへ整形
  3. 指定トークン予算（既定 100M tok）に達するまでシャッフル順に記事を採用
  4. cpt_corpus/{train,val}.jsonl（{"text": ...}）を書き出す

トークン計数は実トークナイザ（Qwen/Qwen3.5-9B-Base、重みは落とさず tokenizer のみ）で行う。
予算は「サブサンプルで run 時間/コストを制御」するためのもの（docs/HANDOFF.md: Modal 無料枠
14h/月に収める）。実 step 時間はスモーク run で較正してから本番 token-budget を確定する。

実行（Mac でローカル整形 → Modal Volume へアップロード）:
  uv run src/build_cpt_corpus.py --out-dir ~/quiz-ai-corpus/corpus --token-budget 100_000_000
  # 疎通: --token-budget 2_000_000 で数千記事だけ通す
  # アップロード: modal volume put quiz-corpus ~/quiz-ai-corpus/corpus/cpt_corpus /cpt_corpus

ライセンス: wiki40b-ja は CC-BY-SA で AI王問題文を含まない（再配布禁止コーパスとは別系統）。
ただし生成物は他コーパスと同じ場所に置くため運用は private で揃える。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

from tqdm import tqdm

DATASET = "range3/wiki40b-ja"
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-9B-Base"

# wiki40b の構造マーカー。content は空白区切りでこれらに挟まれて並ぶ。
_STRUCT_RE = re.compile(r"(_START_ARTICLE_|_START_SECTION_|_START_PARAGRAPH_)")


def clean_wiki40b(text: str) -> str:
    """wiki40b マークアップを素のプレーンテキストへ。

    元: " _START_ARTICLE_ タイトル _START_SECTION_ 概要 _START_PARAGRAPH_ 本文 _NEWLINE_ 本文2 ..."
    後: "タイトル\n\n本文\n本文2\n\n概要\n..."（タイトル→空行→段落、節見出しは空行で区切る）
    _NEWLINE_ は段落内改行なので実改行へ。
    """
    out: list[str] = []
    pieces = _STRUCT_RE.split(text)
    # pieces = [先頭ゴミ, marker, content, marker, content, ...]
    i = 1
    while i < len(pieces):
        marker = pieces[i]
        content = pieces[i + 1] if i + 1 < len(pieces) else ""
        content = content.replace("_NEWLINE_", "\n").strip()
        if content:
            # タイトル/節見出し/段落いずれも「\n\n」結合で区切る（節も段落も同列）。
            out.append(content)
        i += 2
    return "\n\n".join(out).strip()


def load_tokenizer(name: str):
    """重みを落とさず tokenizer のみロード（トークン予算の正確計数用）。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def build(out_dir: str, token_budget: int = 100_000_000,
          tokenizer: str = DEFAULT_TOKENIZER, min_chars: int = 200,
          val_docs: int = 500, seed: int = 3407, limit_scan: int = 0,
          pack_seq: int = 2048) -> dict:
    """wiki40b-ja を整形・サブサンプルし cpt_corpus/{train,val}.jsonl を書く。

    out_dir 直下に cpt_corpus/ を作る。ローカル(uv run)からも Modal の CPU ジョブからも
    同じこの関数を呼ぶ（Modal は out_dir="/data" で Volume へ直接書く）。
    戻り値は統計（採用ブロック数・総トークン・走査数）。

    pack_seq>0（既定2048）: **token粒度の事前パッキング**。記事を tokenize→EOS区切りで連結し、
    ちょうど pack_seq tok のブロックに切って書く。Qwen3.5 は VLM(processor系)で Unsloth の
    sample packing が無効化される（"Sample packing skipped"）ため、ここで詰めて padding 浪費を
    消す＝有効スループットを底上げする。各行 ≒ pack_seq tok の密な系列になる。
    pack_seq=0 で旧来の 1記事1行（packing 検証/比較用）。
    """
    from datasets import load_dataset

    print(f"[cpt] loading {DATASET} train（初回のみ DL ~1.2GB / hf-cache に永続化）…")
    ds = load_dataset(DATASET, split="train")
    ds = ds.shuffle(seed=seed)
    mode = f"pack={pack_seq}tok/block" if pack_seq else "1記事1行(no-pack)"
    print(f"[cpt] 記事数={len(ds)}  token予算={token_budget:,}  tokenizer={tokenizer}  {mode}")

    tok = load_tokenizer(tokenizer)
    eos_id = tok.eos_token_id

    out_root = os.path.join(os.path.expanduser(out_dir), "cpt_corpus")
    os.makedirs(out_root, exist_ok=True)
    train_path = os.path.join(out_root, "train.jsonl")
    val_path = os.path.join(out_root, "val.jsonl")

    blocks: list[str] = []     # 書き出す行（pack時=2048tokブロック / no-pack時=記事）
    buf: list[int] = []        # パッキング用の未確定トークン
    total_tok = 0
    scanned = 0
    pbar = tqdm(ds, desc="scan", unit="doc")
    for ex in pbar:
        scanned += 1
        if limit_scan and scanned > limit_scan:
            break
        text = clean_wiki40b(ex["text"])
        if len(text) < min_chars:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        total_tok += len(ids)
        if pack_seq:
            buf.extend(ids)
            if eos_id is not None:
                buf.append(eos_id)          # 文書境界を学習させる EOS
            while len(buf) >= pack_seq:      # 満ちたら 2048tok ちょうどで切って確定
                blocks.append(tok.decode(buf[:pack_seq]))
                del buf[:pack_seq]
        else:
            blocks.append(text)
        if total_tok >= token_budget:
            break
        if scanned % 2000 == 0:
            pbar.set_postfix(blocks=len(blocks), Mtok=f"{total_tok/1e6:.1f}")

    # 端数バッファ: pack_seq の 1/4 以上あれば最後のブロックとして採用（短すぎは捨てる）
    if pack_seq and len(buf) >= pack_seq // 4:
        blocks.append(tok.decode(buf))

    if len(blocks) <= val_docs:
        raise SystemExit(f"採用ブロック {len(blocks)} が val_docs {val_docs} 以下。"
                         "token-budget を増やすか val-docs を減らせ")

    # シャッフル済みなので末尾を val に回すだけでランダム分割になる
    val = blocks[-val_docs:]
    train = blocks[:-val_docs]

    def dump(path: str, rows: list[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    dump(train_path, train)
    dump(val_path, val)

    mean_tok = total_tok / max(1, len(blocks))
    print(f"[cpt] done. ブロック={len(blocks)}（train={len(train)} / val={len(val)}）  "
          f"総tok={total_tok:,}（mean {mean_tok:.0f} tok/block）  走査={scanned}")
    print(f"[cpt] wrote {train_path}")
    print(f"[cpt] wrote {val_path}")
    return {"kept": len(blocks), "train": len(train), "val": len(val),
            "total_tok": total_tok, "scanned": scanned, "out_root": out_root,
            "pack_seq": pack_seq}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="corpus の親（cpt_corpus/ をこの下に作る）")
    ap.add_argument("--token-budget", type=int, default=100_000_000,
                    help="採用トークン数の上限（既定 100M。run 時間/コスト制御）")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="トークン計数に使う tokenizer（既定=メインLLM の base）")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="この文字数未満の整形後記事はスキップ（スタブ除去）")
    ap.add_argument("--val-docs", type=int, default=500,
                    help="eval_loss 用に末尾から取り分ける val 記事数")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit-scan", type=int, default=0,
                    help=">0 で走査記事数を制限（疎通用）")
    ap.add_argument("--pack-seq", type=int, default=2048,
                    help="token粒度の事前パッキング長（既定2048=cpt max_seq）。0で1記事1行")
    a = ap.parse_args()

    stats = build(a.out_dir, token_budget=a.token_budget, tokenizer=a.tokenizer,
                  min_chars=a.min_chars, val_docs=a.val_docs, seed=a.seed,
                  limit_scan=a.limit_scan, pack_seq=a.pack_seq)
    print(f"[cpt] 次: modal volume put quiz-corpus {stats['out_root']} /cpt_corpus")


if __name__ == "__main__":
    main()
