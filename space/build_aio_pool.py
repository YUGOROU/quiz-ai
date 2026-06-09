# /// script
# requires-python = ">=3.11"
# ///
"""cl-tohoku/quiz-datasets の data/aio（CC BY-SA 4.0・キュービック/カプリティオ作成）から
デモ用問題プール questions_pool_ja.json を生成する。再配布可能＝ハッカソンで共有OK。

  uv run space/build_aio_pool.py
"""
import json
import re
import urllib.request

BASE = "https://raw.githubusercontent.com/cl-tohoku/quiz-datasets/main/data/aio"
FILES = ["aio_01_dev1.txt", "aio_01_dev2.txt", "aio_01_test_lb.txt",
         "aio_01_test_lc.txt", "aio_01_unused.txt", "aio_02_dev.txt"]


def main():
    seen_q = set()
    out = []
    for fn in FILES:
        raw = urllib.request.urlopen(f"{BASE}/{fn}").read().decode("utf-8")
        lines = raw.splitlines()
        header = lines[0].split("\t")
        qi = header.index("original_question")
        ai = header.index("original_answer")
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) <= max(qi, ai):
                continue
            q = cols[qi].strip()
            a = cols[ai].strip()
            # 表記ゆれ・別解は「_」区切りのことがある → 先頭を主正解に。
            a_main = a.split("_")[0].strip()
            if not q or not a_main or len(q) < 20 or len(a_main) < 1:
                continue
            if a_main in q:            # 答えが問題文に出ている＝早押しにならない
                continue
            if q in seen_q:
                continue
            seen_q.add(q)
            out.append({"id": len(out) + 1, "full": q, "truth": a_main})

    data = {"match": "早押しクイズAI デモマッチ", "lang": "ja",
            "note": "AI王 data/aio (cl-tohoku/quiz-datasets) 由来・CC BY-SA 4.0（キュービック/カプリティオ作成）＝再配布可。",
            "license": "CC BY-SA 4.0", "questions": out}
    with open("space/questions_pool_ja.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"[aio] {len(out)} 問を questions_pool_ja.json に書き出し")
    print("sample:", json.dumps(out[0], ensure_ascii=False))
    print("sample:", json.dumps(out[len(out) // 2], ensure_ascii=False))


if __name__ == "__main__":
    main()
