# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pykakasi",
# ]
# ///
"""問題プールの正解（truth）に読み仮名 truth_kana を一括付与する。

目的（UX 修正 A4）: frontend の解答判定はかな⇄カナ統一のみで漢字⇄読みを救済できず、
「夏目漱石」が正解の問題に「なつめそうせき」とタイプすると誤判定になる。
pykakasi（辞書ベース・オフライン）で truth のひらがな読みを生成して JSON に持たせ、
frontend judge は truth と truth_kana の両方に照合する。

使い方:
  uv run space/add_kana.py            # questions_pool_ja.json / questions_ja.json を更新
"""
from __future__ import annotations

import json
import pathlib

import pykakasi

HERE = pathlib.Path(__file__).resolve().parent
TARGETS = ["questions_pool_ja.json", "questions_ja.json"]

kks = pykakasi.kakasi()


def to_hira(text: str) -> str:
    return "".join(item["hira"] for item in kks.convert(text))


def main() -> None:
    for name in TARGETS:
        path = HERE / name
        if not path.exists():
            print(f"[add_kana] skip (not found): {name}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        qs = data.get("questions", [])
        added = same = 0
        for q in qs:
            truth = q.get("truth", "")
            t = truth[0] if isinstance(truth, list) else truth
            kana = to_hira(t)
            # 読みが元と同じ（既にかな/英数）なら冗長なので持たせない。
            if kana and kana != t:
                q["truth_kana"] = kana
                added += 1
            else:
                q.pop("truth_kana", None)
                same += 1
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
        print(f"[add_kana] {name}: {len(qs)} questions, kana added {added}, unchanged {same}")


if __name__ == "__main__":
    main()
