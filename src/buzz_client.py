# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Phase 0b: buzz サーバ（serve/serve_buzz.py）を叩く buzz_decider ファクトリ。

orchestrator.run_episode(..., buzz_decider=...) に渡す同期 callable を返す。
serve は localhost 同居前提（同一 Vast 箱）なので urllib で十分高速・依存ゼロ。

  buzz_decider(pos, prefix) -> bool   # conf >= θ_dynamic で True（=buzz確定）

θ_dynamic は CLAUDE.md の設計どおりオーケストレータ側で外付け調整する
（base_θ × game_state_factor）。ここでは base_θ を受け取り、必要なら呼び出し側で
スケールした θ を make_buzz_decider に渡す。

実効打点メモ（HANDOFF 2026-06-07 θトレードオフ実測）:
  S-buzz ちょうど(=θ低め)だと main 正解率 ~62%。θ を上げて打点を +4〜8字 遅らせると
  main 69〜74% でメイン側ゲート(≥65%)を超える。E2E では θ で速度↔精度を調整する。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable


def query_conf(base_url: str, prefix: str, n: int | None = None,
               timeout: float = 1.0) -> float:
    """serve に prefix を投げて conf=sigmoid(logit) を取得する。"""
    payload = json.dumps({"prefix": prefix, "n": n}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/conf", data=payload,
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return float(json.loads(resp.read())["conf"])


def make_buzz_decider(
    base_url: str = "http://127.0.0.1:8001",
    theta: float = 0.45,
    timeout: float = 1.0,
    on_error: str = "no-buzz",   # serve 不通時の挙動: "no-buzz"（安全側）/"raise"
) -> Callable[[int, str], bool]:
    """conf >= theta で buzz する同期 decider を返す。

    theta は buzz 回帰ヘッドの best θ（純回帰ヘッド=0.45）を既定とする。打点を
    遅らせて精度を上げたい場合は theta を上げる（=より自信が出るまで待つ）。
    """
    def decider(pos: int, prefix: str) -> bool:
        try:
            conf = query_conf(base_url, prefix, n=pos, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if on_error == "raise":
                raise
            return False  # serve 不通 → buzz しない（誤爆より取りこぼし側に倒す）
        return conf >= theta

    return decider


def healthcheck(base_url: str = "http://127.0.0.1:8001", timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    # 簡易疎通確認: serve 起動済みの箱で `uv run src/buzz_client.py "<prefix>"`
    import sys

    base = "http://127.0.0.1:8001"
    if not healthcheck(base):
        print(f"[buzz-client] serve に到達できません: {base}")
        sys.exit(1)
    prefix = sys.argv[1] if len(sys.argv) > 1 else "743年に聖武天皇が出した"
    conf = query_conf(base, prefix)
    print(f"conf={conf:.3f}  buzz@θ0.45={conf >= 0.45}")
