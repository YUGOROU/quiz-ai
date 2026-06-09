# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "tqdm",
# ]
# ///
"""questions_pool_ja.json の各問にジャンルラベルを付与する（LLMバッチ分類）。

hpprc/quiz-works はカテゴリ情報を持たないため、20種類の固定タクソノミーに
LLM（Crof.ai deepseek-v4-flash・OpenAI互換）で分類する。Space の「得意ジャンル選択」用。

- 中断再開: ラベルは cache(jsonl) に id 単位で逐次 append。再起動時に既処理 id を読み込む。
- 並列: Crof.ai は高並列で一時BANされた実績があるため**控えめ**（既定 workers=2）。
        さらに保守的に行くなら --workers 1（完全逐次）＋ --delay 0.3 など。
- APIキーは環境変数（CROFAI_API_KEY / CROFAI_BASE_URL）から直接読む。表示しない。

実行（キーはユーザー環境で・露出させない）:
  CROFAI_API_KEY=<key> CROFAI_BASE_URL=https://crof.ai/v1 uv run src/label_genres.py
  ... --workers 1 --delay 0.3   # さらに控えめに
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qutils  # noqa: E402

# 20ジャンル（競技クイズ標準ベース・ユーザー確定 2026-06-08）。
GENRES = [
    "文学・言葉", "歴史", "地理", "政治・経済", "社会・時事",
    "理科・科学", "数学", "医学・人体", "スポーツ", "野球",
    "音楽", "美術・建築", "映画・演劇", "アニメ・漫画・ゲーム", "アイドル・芸能",
    "テレビ・お笑い", "グルメ・料理", "生活・雑学", "動物・植物", "国際・宗教・神話",
]
GENRE_SET = set(GENRES)
FALLBACK = "生活・雑学"  # 分類不能・不正出力時のデフォルト

_PROMPT = (
    "あなたは早押しクイズの問題を1つのジャンルに分類する分類器です。\n"
    "次のジャンルのうち、最も適切なものを**1つだけ**、表記そのままで出力してください。\n"
    "説明や記号は不要。ジャンル名のみを1行で答えてください。\n\n"
    "ジャンル一覧:\n" + "\n".join(f"- {g}" for g in GENRES) + "\n\n"
    "判断基準の補足:\n"
    "- 野球に関する問題は「スポーツ」でなく「野球」。\n"
    "- 物理・化学・生物・天文・地学は「理科・科学」。人体・病気・薬は「医学・人体」。\n"
    "- 漢字・語源・ことわざ・小説・詩は「文学・言葉」。\n"
    "- 歌手・楽曲・楽器・作曲家は「音楽」。アイドルグループは「アイドル・芸能」。\n"
    "- 神話・宗教・海外の地理以外の国際事情は「国際・宗教・神話」。\n\n"
    "問題: {q}\n答え: {a}\n\nジャンル:"
)


def _classify(client, model, q: str, a: str) -> str:
    out = qutils.chat(
        client, model, _PROMPT.format(q=q, a=a),
        max_tokens=16, temperature=0.0, reasoning_effort="none",
    )
    if not out:
        return FALLBACK
    out = out.strip().splitlines()[0].strip().strip("「」 　-・:：")
    if out in GENRE_SET:
        return out
    # 部分一致で救済（モデルが微妙に違う表記を返した場合）。
    for g in GENRES:
        if g in out or out in g:
            return g
    return FALLBACK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="space/questions_pool_ja.json")
    ap.add_argument("--cache", default="space/.genre_cache.jsonl")
    ap.add_argument("--workers", type=int, default=2)   # 控えめ（前回の一時BAN対策）
    ap.add_argument("--delay", type=float, default=0.15)  # 各リクエスト前の軽いディレイ(s)
    args = ap.parse_args()

    inp = pathlib.Path(args.inp)
    cache = pathlib.Path(args.cache)
    data = json.loads(inp.read_text(encoding="utf-8"))
    questions = data["questions"]

    # 既処理 id を cache から読む（中断再開）。
    done: dict[int, str] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done[r["id"]] = r["genre"]
    todo = [q for q in questions if q["id"] not in done]
    print(f"[label] total={len(questions)} done={len(done)} todo={len(todo)}")

    if todo:
        client, model = qutils.get_client()
        print(f"[label] model={model} workers={args.workers}")
        from tqdm import tqdm
        lock = threading.Lock()
        cf = cache.open("a", encoding="utf-8")

        import time

        def work(q):
            if args.delay:
                time.sleep(args.delay)
            g = _classify(client, model, q["full"], q["truth"])
            with lock:
                cf.write(json.dumps({"id": q["id"], "genre": g}, ensure_ascii=False) + "\n")
                cf.flush()
            done[q["id"]] = g
            return g

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, q) for q in todo]
            for _ in tqdm(as_completed(futs), total=len(futs)):
                pass
        cf.close()

    # 元 JSON に genre を反映して書き戻し。
    for q in questions:
        q["genre"] = done.get(q["id"], FALLBACK)
    inp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # 分布レポート。
    from collections import Counter
    dist = Counter(q["genre"] for q in questions)
    print("\n[label] ジャンル分布:")
    for g in GENRES:
        print(f"  {g:16} {dist.get(g, 0):4}")
    print(f"\n[label] 反映完了 → {inp}")


if __name__ == "__main__":
    main()
