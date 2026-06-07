# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "torch",
#   "transformers",
#   "huggingface_hub",
# ]
# ///
"""Phase 0b: buzz 回帰ヘッド（割り込みモデル）の推論サーバ。

`train/buzz_reg.py` の BuzzRegressor（LFM2.5 backbone + 最終非padトークン
プーリング + Linear(h,1)）を HF repo から復元し、prefix → conf=sigmoid(logit)
を返す HTTP サーバ。orchestrator の buzz_decider は localhost のこの API を叩く。

設計メモ:
  - 入力整形は学習時と厳密一致させる（USER_TEMPLATE）。ズレると conf がずれる。
  - vLLM は回帰ヘッドを serve できない（生成モデル前提）ため自前 FastAPI。
  - メイン(gemma) の vLLM serve とは別プロセス・別ポートで同一GPUに同居する。
    バックボーン 1.2B bf16 は VRAM ~3GB（5090 32GB で gemma 4bit と共存可）。

実行（Vast 5090 上）:
  export HF_TOKEN=hf_...
  uv run serve/serve_buzz.py --repo YUGOROU/quiz-buzz-reg-1.2bjp-merged --port 8001

ヘルスチェック / 推論:
  curl localhost:8001/health
  curl -s localhost:8001/conf -H 'content-type: application/json' \
       -d '{"prefix": "743年に聖武天皇が出した"}'
  → {"conf": 0.83, "n": 12}
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# 学習時（corpus-1 / train/buzz_reg.py）と厳密一致させること。
USER_TEMPLATE = "問題文（{n}文字目まで）:\n{prefix}"
HEAD_FILE = "buzz_head.pt"


class BuzzRegressor(nn.Module):
    """train/buzz_reg.py と同一構造（backbone + 最終非padトークンプーリング + Linear）。"""

    def __init__(self, backbone, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)
        self.head.to(dtype=next(backbone.parameters()).dtype)

    def forward(self, input_ids=None, attention_mask=None, **_):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state                       # [B, T, H]
        last = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)  # 右パディング前提
        pooled = hs[torch.arange(hs.size(0), device=hs.device), last]
        return self.head(pooled).squeeze(-1)             # [B] 生スコア


def load_regressor(repo: str, max_seq_length: int, token: str | None, device: str):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = max_seq_length
    backbone = AutoModel.from_pretrained(repo, dtype="bfloat16", token=token)
    model = BuzzRegressor(backbone, backbone.config.hidden_size)
    head_path = hf_hub_download(repo, HEAD_FILE, token=token)
    model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.head.to(dtype=next(backbone.parameters()).dtype)
    model.to(device).eval()
    return model, tok


class ConfRequest(BaseModel):
    prefix: str
    n: int | None = None     # 省略時は len(prefix)。学習は n=文字数を使う。


class ConfBatchRequest(BaseModel):
    prefixes: list[str]


def build_app(model, tok, max_seq_length: int, device: str) -> FastAPI:
    app = FastAPI(title="quiz-buzz")

    @torch.no_grad()
    def _conf(texts: list[str]) -> list[float]:
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_seq_length).to(device)
        logits = model(**enc)
        return torch.sigmoid(logits.float()).reshape(-1).tolist()

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device}

    @app.post("/conf")
    def conf(req: ConfRequest):
        n = req.n if req.n is not None else len(req.prefix)
        text = USER_TEMPLATE.format(n=n, prefix=req.prefix)
        return {"conf": _conf([text])[0], "n": n}

    @app.post("/conf_batch")
    def conf_batch(req: ConfBatchRequest):
        texts = [USER_TEMPLATE.format(n=len(p), prefix=p) for p in req.prefixes]
        return {"confs": _conf(texts)}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="YUGOROU/quiz-buzz-reg-1.2bjp-merged")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-seq-length", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_TOKEN")
    print(f"[buzz-serve] loading {args.repo} on {device} …")
    model, tok = load_regressor(args.repo, args.max_seq_length, token, device)
    print(f"[buzz-serve] ready. listening on {args.host}:{args.port}")
    app = build_app(model, tok, args.max_seq_length, device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
