"""割り込みモデル(buzz)の回帰ヘッド版 SFT＋評価 — 自己完結 Modal アプリ。

背景（docs/HANDOFF.md・メモリ buzz-mae-not-capacity-bound）:
  CLM で `<think>…</think>{confidence}` を文字列生成する方式は、buzz位置 MAE≤8 の
  移行ゲートに届かない（350M 10.55 → 1.2B 9.96・容量律速でない）。ボトルネックは
  buzz周辺の confidence 回帰の当てはまり（位置誤差 ≒ conf誤差 / カーブ斜面）。
  本スクリプトは quiz-ai.md Phase0 フォールバックの「回帰ヘッド化」を実装し、
  confidence を直接回帰最適化する（B を直接叩く）。

設計:
  - バックボーン = LFM2.5（AutoModel・Lfm2Model）を**全パラFT**（LoRA被覆が薄いため
    full-FT が素直＝メモリ lfm2-lora-undercoverage）。LFM2 は ForSequenceClassification を
    持たない（5.5.0/5.10.2 で未提供）ため、最終非padトークン hidden → Linear(h,1) の
    自前ヘッドを乗せる。
  - 損失 = soft-target BCEWithLogits（ラベル = corpus-1 の meta.confidence_label ∈[0,1]）。
    推論 conf = sigmoid(logit)。CLM の `<think>` 出力は使わない。
  - 入力 = corpus-1 の user content をそのまま（"問題文（{n}文字目まで）:\\n{prefix}"）。
    chat テンプレートは付けない（回帰エンコーダなので train/eval の整形一致だけが要件）。

データは HF private データセット `YUGOROU/quiz-ai-corpus`（sft_corpus_1_steep/ ・
annotated_questions.jsonl）から取得。

実行（train）:
  uv run --with modal modal run train/buzz_reg.py::train_main \
    --base-model LiquidAI/LFM2.5-1.2B-JP-202606 --data-subdir sft_corpus_1_steep \
    --push-to-hub YUGOROU/quiz-buzz-reg-1.2bjp --gpu A100-40GB

実行（eval・移行ゲート MAE≤8 判定）:
  uv run --with modal modal run train/buzz_reg.py::eval_main \
    --model-repo YUGOROU/quiz-buzz-reg-1.2bjp --n-questions 200
"""
import os

import modal

app = modal.App("quiz-buzz-reg")

_HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "transformers", "datasets", "accelerate", "bitsandbytes",
        "huggingface_hub", "hf_transfer", "nvidia-ml-py",
        "sentencepiece", "protobuf",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]

# build_corpus1.py / eval_buzz.py と完全一致させること（不一致は評価を無効化する）。
USER_TEMPLATE = "問題文（{n}文字目まで）:\n{prefix}"
HEAD_FILE = "buzz_head.pt"   # 回帰ヘッドの state_dict をこのファイル名で配布する

# torch を使うクラス定義は「torch がある環境（=Modal コンテナ）」でだけ評価する。
# local_entrypoint は Mac で走り torch 不在なので、module top で import すると落ちる。
try:
    import torch
    import torch.nn as nn

    class BuzzRegressor(nn.Module):
        """LFM2.5 backbone + 最終非padトークンプーリング + Linear(h,1) 回帰ヘッド。

        forward は HF Trainer 互換で {"loss", "logits"} を返す。logits は生スコア
        （推論 conf = sigmoid(logits)）。labels はソフトターゲット ∈[0,1]。
        """

        def __init__(self, backbone, hidden_size: int):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(hidden_size, 1)
            self.head.to(dtype=next(backbone.parameters()).dtype)

        def forward(self, input_ids=None, attention_mask=None, labels=None, **_):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hs = out.last_hidden_state                       # [B, T, H]
            # 右パディング前提: 各系列の最終非padトークン位置 = mask 和 - 1
            last = attention_mask.long().sum(dim=1) - 1      # [B]
            last = last.clamp(min=0)
            pooled = hs[torch.arange(hs.size(0), device=hs.device), last]  # [B, H]
            logits = self.head(pooled).squeeze(-1)           # [B]
            loss = None
            if labels is not None:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), labels.float())
            return {"loss": loss, "logits": logits}

    def _load_regressor(model_repo: str, max_seq_length: int, token: str | None):
        """HF repo から backbone(AutoModel) + 回帰ヘッドを復元して BuzzRegressor を返す。"""
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download

        tok = AutoTokenizer.from_pretrained(model_repo, token=token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        tok.model_max_length = max_seq_length
        backbone = AutoModel.from_pretrained(model_repo, dtype="bfloat16", token=token)
        model = BuzzRegressor(backbone, backbone.config.hidden_size)
        head_path = hf_hub_download(model_repo, HEAD_FILE, token=token)
        model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        model.head.to(dtype=next(backbone.parameters()).dtype)
        return model, tok

    @torch.no_grad()
    def _predict_conf(model, tok, texts, batch_size, max_seq_length, device="cuda"):
        """user content のリスト → conf=sigmoid(logit) のリスト。バッチ forward。"""
        confs = []
        model.eval()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_seq_length).to(device)
            logits = model(**enc)["logits"]
            confs.extend(torch.sigmoid(logits.float()).tolist())
        return confs

    # --- デュアルヘッド(A): LM head(reasoning生成) + 回帰ヘッド(post-think calibration) ---
    class BuzzDualHead(nn.Module):
        """LFM2.5 CausalLM（LM head=reasoning生成）＋ 回帰ヘッド（post-think hidden で calibration）。

        損失 = CLM(response-only) ＋ reg_weight·BCE。推論(A)は <think>…</think> を生成し、
        その最終トークンの hidden を回帰ヘッドで読んで conf とする（reasoning を意思決定の
        計算に組み込みつつ、判定は滑らかな連続スカラーで読む）。forward は HF Trainer 互換。
        """

        def __init__(self, causal_lm, hidden_size: int, reg_weight: float = 3.0):
            super().__init__()
            self.lm = causal_lm
            self.head = nn.Linear(hidden_size, 1)
            self.head.to(dtype=next(causal_lm.parameters()).dtype)
            self.reg_weight = reg_weight

        def forward(self, input_ids=None, attention_mask=None, labels=None,
                    conf_labels=None, pool_idx=None, **_):
            # 省メモリ: output_hidden_states（全層保持）を避け、decoder の最終 hidden だけ取り
            # lm_head でロジットを手動計算（全語彙ロジット [B,T,V] が最大コスト）。
            hs = self.lm.get_decoder()(
                input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            loss = None
            if labels is not None:
                logits = self.lm.get_output_embeddings()(hs)      # [B, T, V]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1), ignore_index=-100)
            logits_reg = None
            if pool_idx is not None:
                pooled = hs[torch.arange(hs.size(0), device=hs.device), pool_idx]
                logits_reg = self.head(pooled).squeeze(-1)
                if conf_labels is not None and loss is not None:
                    reg = nn.functional.binary_cross_entropy_with_logits(
                        logits_reg.float(), conf_labels.float())
                    loss = loss + self.reg_weight * reg
            return {"loss": loss, "logits": logits_reg}

    def _load_dual(model_repo: str, max_seq_length: int, token: str | None,
                   reg_weight: float = 3.0):
        """HF repo から CausalLM + 回帰ヘッドを復元して BuzzDualHead を返す。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download

        tok = AutoTokenizer.from_pretrained(model_repo, token=token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.model_max_length = max_seq_length
        lm = AutoModelForCausalLM.from_pretrained(model_repo, dtype="bfloat16", token=token)
        model = BuzzDualHead(lm, lm.config.hidden_size, reg_weight)
        head_path = hf_hub_download(model_repo, HEAD_FILE, token=token)
        model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        model.head.to(dtype=next(lm.parameters()).dtype)
        return model, tok

    @torch.no_grad()
    def _predict_conf_dual(model, tok, user_contents, batch_size, max_seq_length,
                           max_think_tokens=48, device="cuda"):
        """(A) 各 prefix で <think>…</think> を生成 → その最終 hidden を回帰ヘッドで読む。
        生成は left padding、pooling forward は right padding（訓練の pool 位置と一致）。"""
        model.eval()
        confs = []
        for i in range(0, len(user_contents), batch_size):
            batch = user_contents[i:i + batch_size]
            tok.padding_side = "left"
            enc = tok([c + "\n" for c in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_seq_length).to(device)
            gen = model.lm.generate(**enc, max_new_tokens=max_think_tokens,
                                    do_sample=False, pad_token_id=tok.pad_token_id)
            cont = gen[:, enc["input_ids"].shape[1]:]      # 生成ぶんのみ（left-pad で列固定）
            full_texts = []
            for c, row in zip(batch, cont):
                t = tok.decode(row, skip_special_tokens=True)
                think = t.split("</think>")[0] + "</think>" if "</think>" in t else t
                full_texts.append(c + "\n" + think)
            tok.padding_side = "right"
            enc2 = tok(full_texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_seq_length).to(device)
            hs = model.lm(**enc2, output_hidden_states=True).hidden_states[-1]
            last = enc2["attention_mask"].long().sum(1) - 1
            pooled = hs[torch.arange(hs.size(0), device=hs.device), last]
            logit = model.head(pooled).squeeze(-1)
            confs.extend(torch.sigmoid(logit.float()).tolist())
        return confs

except ImportError:    # Mac の local_entrypoint 評価時（torch 不在）はスキップ
    pass


# ============================================================
# 評価指標（A: buzz位置MAE / B: conf回帰 / C: 全文conf）
#   eval_buzz.py の閾値交差・θスイープと同一ロジック。conf は回帰ヘッドから直接得る。
# ============================================================
def _buzz_metrics(qids, curves, annot, threshold):
    """curves={qid:[(pos,conf)...]} と buzz_char から A の誤差統計を返す。"""
    import statistics

    def buzz_errors(th):
        errs, nc = [], 0
        for qid in qids:
            pts = sorted(curves.get(qid, []))
            if not pts:
                continue
            buzz_char = annot[qid]["buzz_char"]
            L = annot[qid]["question_length"]
            pred, prev = None, None
            for pos, c in pts:
                if c >= th:
                    if prev is None:
                        pred = pos
                    else:
                        p0, c0 = prev
                        frac = (th - c0) / (c - c0) if c != c0 else 0.0
                        pred = p0 + frac * (pos - p0)
                    break
                prev = (pos, c)
            if pred is None:
                nc += 1
                pred = L
            errs.append(abs(pred - buzz_char))
        return errs, nc

    a_errs, no_cross = buzz_errors(threshold)
    mae_a = statistics.mean(a_errs) if a_errs else float("nan")
    median_a = statistics.median(a_errs) if a_errs else float("nan")
    within8 = sum(1 for e in a_errs if e <= 8) / len(a_errs) if a_errs else float("nan")

    sweep_th = [0.30 + 0.05 * i for i in range(9)]  # 0.30..0.70
    th_results = []
    for th in sweep_th:
        e, nc = buzz_errors(th)
        if e:
            th_results.append((th, statistics.mean(e),
                               sum(1 for x in e if x <= 8) / len(e), nc))
    best = min(th_results, key=lambda r: r[1]) if th_results else None
    return dict(mae=mae_a, median=median_a, within8=within8, no_cross=no_cross,
                n=len(a_errs), th_results=th_results, best=best)


# ============================================================
# 訓練
# ============================================================
@app.function(
    image=image, gpu="A100-40GB", cpu=8.0, timeout=60 * 60 * 6,
    volumes={"/hf": hf_cache}, secrets=secrets,
)
def train(base_model: str, data_subdir: str, dataset_repo: str,
          push_to_hub: str | None, hub_private: bool,
          epochs: float, batch_size: int, grad_accum: int,
          learning_rate: float, max_seq_length: int, warmup_ratio: float,
          max_steps: int, seed: int, grad_ckpt: bool):
    import json

    import torch
    from transformers import (AutoModel, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    set_seed(seed)

    # push 指定時は訓練前に write 権限を fail-fast 検証（長時間 run 後の push 失敗を防ぐ）
    if push_to_hub:
        who = HfApi().whoami(token=token)
        role = who.get("auth", {}).get("accessToken", {}).get("role")
        print(f"[reg] HF token: name={who.get('name')} role={role}")
        if role != "write":
            raise SystemExit("HF_TOKEN に write 権限なし。Modal Secret 'huggingface' を更新せよ")

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(base_model, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"     # 最終非padトークンプーリングのため右パディング
    tok.model_max_length = max_seq_length

    # --- backbone + 回帰ヘッド ---
    backbone = AutoModel.from_pretrained(base_model, dtype="bfloat16", token=token)
    backbone.config.use_cache = False        # 訓練は生成しない（cache不要）
    # gradient checkpointing は既定 OFF。LFM2.5-350M/1.2B は小さく activation も短seqで
    # 極小（bs64 でも peak ~9GiB）＝メモリ非律速なので、再計算ぶんの FLOPs を払うのは
    # 純損。メモリが逼迫する超大バッチ時のみ --grad-ckpt で有効化する。
    if grad_ckpt:
        backbone.gradient_checkpointing_enable()
    model = BuzzRegressor(backbone, backbone.config.hidden_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[reg] base={base_model}  hidden={backbone.config.hidden_size}  "
          f"params={n_params/1e6:.0f}M  full-FT  loss=soft-BCE")

    # --- データ（corpus-1 の user content と meta.confidence_label） ---
    def load_split(split):
        path = hf_hub_download(dataset_repo, f"{data_subdir}/{split}.jsonl",
                               repo_type="dataset", token=token)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        return [{"text": r["messages"][0]["content"],
                 "label": float(r["meta"]["confidence_label"])} for r in rows]

    train_rows = load_split("train")
    val_rows = load_split("val")
    print(f"[reg] data={dataset_repo}/{data_subdir}  train={len(train_rows)} val={len(val_rows)}")

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        enc = tok([b["text"] for b in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_seq_length)
        enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        return enc

    def compute_metrics(eval_pred):
        import statistics
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + torch.exp(-torch.tensor(logits).float()))
        labels = torch.tensor(labels).float()
        mae = (probs - labels).abs().mean().item()
        rmse = ((probs - labels) ** 2).mean().sqrt().item()
        return {"conf_mae": mae, "conf_rmse": rmse}

    smoke = max_steps > 0
    args = TrainingArguments(
        output_dir="/tmp/buzz_reg_out",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps if smoke else -1,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        logging_steps=10 if not smoke else 1,
        save_strategy="no",            # 保存はヘッド込みで手動（カスタム nn.Module のため）
        eval_strategy="no" if smoke else "epoch",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,   # dict バッチを Trainer に間引かせない
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=DS(train_rows), eval_dataset=DS(val_rows),
        data_collator=collate, compute_metrics=compute_metrics,
    )

    stats = trainer.train()
    if not smoke:
        metrics = trainer.evaluate()
        print(f"[reg] val: conf_MAE={metrics.get('eval_conf_mae'):.4f}  "
              f"conf_RMSE={metrics.get('eval_conf_rmse'):.4f}")

    peak = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    m = stats.metrics
    print(f"[reg] peak GPU {peak:.1f}/{total:.0f} GiB  runtime={m.get('train_runtime')}s  "
          f"train_loss={m.get('train_loss')}")

    # --- 保存（backbone は save_pretrained・ヘッドは state_dict 別ファイル） ---
    out_dir = "/tmp/buzz_reg_merged"
    model.backbone.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), os.path.join(out_dir, HEAD_FILE))
    print(f"[reg] saved backbone+{HEAD_FILE} -> {out_dir}")

    if push_to_hub:
        # ⚠ ライセンス: コーパス由来の重み。既定 Private。
        repo = f"{push_to_hub}-merged"
        api = HfApi()
        api.create_repo(repo, private=hub_private, exist_ok=True, token=token)
        api.upload_folder(folder_path=out_dir, repo_id=repo, token=token)
        print(f"[reg] pushed ({'private' if hub_private else 'public'}) -> {repo}")
    hf_cache.commit()


# ============================================================
# 訓練（デュアルヘッド(A): CLM reasoning ＋ 回帰ヘッド calibration の同時学習）
# ============================================================
@app.function(
    image=image, gpu="A100-40GB", cpu=8.0, timeout=60 * 60 * 6,
    volumes={"/hf": hf_cache}, secrets=secrets,
)
def train_dual(base_model: str, data_subdir: str, dataset_repo: str,
               push_to_hub: str | None, hub_private: bool,
               epochs: float, batch_size: int, grad_accum: int,
               learning_rate: float, max_seq_length: int, warmup_ratio: float,
               max_steps: int, seed: int, grad_ckpt: bool, reg_weight: float):
    import json
    import statistics

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    set_seed(seed)
    if push_to_hub:
        who = HfApi().whoami(token=token)
        role = who.get("auth", {}).get("accessToken", {}).get("role")
        print(f"[dual] HF token: name={who.get('name')} role={role}")
        if role != "write":
            raise SystemExit("HF_TOKEN に write 権限なし。Modal Secret 'huggingface' を更新せよ")

    tok = AutoTokenizer.from_pretrained(base_model, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = max_seq_length

    lm = AutoModelForCausalLM.from_pretrained(base_model, dtype="bfloat16", token=token)
    lm.config.use_cache = False
    if grad_ckpt:
        lm.gradient_checkpointing_enable()
    model = BuzzDualHead(lm, lm.config.hidden_size, reg_weight)
    print(f"[dual] base={base_model}  hidden={lm.config.hidden_size}  "
          f"reg_weight={reg_weight}  loss=CLM+λ·BCE  full-FT")

    # データ: prompt=user+"\n"（マスク）/ resp=<think>…</think>（CLM対象・conf文字は捨てる）/
    #         conf=meta.confidence_label（回帰ヘッド target）。
    def load_split(split):
        path = hf_hub_download(dataset_repo, f"{data_subdir}/{split}.jsonl",
                               repo_type="dataset", token=token)
        rows = []
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            asst = r["messages"][1]["content"]
            think = asst.split("</think>")[0] + "</think>" if "</think>" in asst else asst
            rows.append({"prompt": r["messages"][0]["content"] + "\n", "resp": think,
                         "conf": float(r["meta"]["confidence_label"])})
        return rows

    train_rows = load_split("train")
    val_rows = load_split("val")
    print(f"[dual] data={dataset_repo}/{data_subdir}  train={len(train_rows)} val={len(val_rows)}")

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        # prompt をマスク（-100）、resp のみ CLM 対象。pool_idx = resp 末尾（=</think>末尾）。
        seqs, labs, pool_idx, confs = [], [], [], []
        for b in batch:
            p = tok(b["prompt"], add_special_tokens=True).input_ids
            r = tok(b["resp"], add_special_tokens=False).input_ids
            ids = (p + r)[:max_seq_length]
            lab = ([-100] * len(p) + r)[:max_seq_length]
            seqs.append(ids)
            labs.append(lab)
            pool_idx.append(len(ids) - 1)
            confs.append(b["conf"])
        maxlen = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        input_ids, attn, labels = [], [], []
        for s, lab in zip(seqs, labs):
            n = maxlen - len(s)
            input_ids.append(s + [pad] * n)          # 右パディング（pool_idx と整合）
            attn.append([1] * len(s) + [0] * n)
            labels.append(lab + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels),
            "conf_labels": torch.tensor(confs, dtype=torch.float32),
            "pool_idx": torch.tensor(pool_idx, dtype=torch.long),
        }

    smoke = max_steps > 0
    args = TrainingArguments(
        output_dir="/tmp/buzz_dual_out",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs, max_steps=max_steps if smoke else -1,
        learning_rate=learning_rate, warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.01,
        bf16=True, logging_steps=10 if not smoke else 1,
        save_strategy="no", eval_strategy="no",   # val は手動（teacher-forced conf MAE）
        dataloader_num_workers=4, dataloader_pin_memory=True,
        report_to="none", seed=seed, remove_unused_columns=False,
        label_names=["labels", "conf_labels"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=DS(train_rows),
                      data_collator=collate)
    stats = trainer.train()

    # 手動 val（teacher-forced: true think を入れ post-think hidden の conf MAE）
    if not smoke:
        model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(val_rows), batch_size):
                vb = {k: v.to("cuda") for k, v in collate(val_rows[i:i + batch_size]).items()}
                p = torch.sigmoid(model(**vb)["logits"].float())
                errs.extend((p - vb["conf_labels"]).abs().tolist())
        print(f"[dual] val teacher-forced conf_MAE={statistics.mean(errs):.4f} (n={len(errs)})")

    peak = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    m = stats.metrics
    print(f"[dual] peak GPU {peak:.1f}/{total:.0f} GiB  runtime={m.get('train_runtime')}s  "
          f"train_loss={m.get('train_loss')}")

    out_dir = "/tmp/buzz_dual_merged"
    model.lm.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), os.path.join(out_dir, HEAD_FILE))
    print(f"[dual] saved CausalLM+{HEAD_FILE} -> {out_dir}")
    if push_to_hub:
        repo = f"{push_to_hub}-merged"
        api = HfApi()
        api.create_repo(repo, private=hub_private, exist_ok=True, token=token)
        api.upload_folder(folder_path=out_dir, repo_id=repo, token=token)
        print(f"[dual] pushed ({'private' if hub_private else 'public'}) -> {repo}")
    hf_cache.commit()


# ============================================================
# 評価（移行ゲート MAE≤8 判定・回帰ヘッドから conf を直接取得）
# ============================================================
# --- 評価の共有ヘルパー（reg / dual で共通・torch 不要の純 stats） ---------
def _load_eval_data(dataset_repo, conf_split, token):
    """conf_split="<subdir>:<split>"（既定 sft_corpus_1_steep:test）から test_rows と annot。"""
    import json
    from huggingface_hub import hf_hub_download
    subdir, _, split = conf_split.partition(":")
    if not split:
        subdir, split = "sft_corpus_1_steep", subdir
    test_path = hf_hub_download(dataset_repo, f"{subdir}/{split}.jsonl",
                               repo_type="dataset", token=token)
    annot_path = hf_hub_download(dataset_repo, "annotated_questions.jsonl",
                                repo_type="dataset", token=token)
    test_rows = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    annot = {}
    for l in open(annot_path, encoding="utf-8"):
        q = json.loads(l)
        annot[q["qid"]] = q
    return test_rows, annot, subdir, split


def _build_sweep(test_rows, annot, n_questions, sweep_points):
    """A/C 用に test qid を dense sweep し (qids, sweep_jobs, sweep_prompts) を返す。"""
    import random
    test_qids = {r["meta"]["qid"] for r in test_rows if r["meta"].get("qid") in annot}
    qids = sorted(test_qids)
    random.Random(3407).shuffle(qids)
    qids = qids[:n_questions]
    sweep_jobs = []
    for qid in qids:
        L = annot[qid]["question_length"]
        start = max(10, int(0.15 * L))
        if start >= L:
            start = max(1, L - 1)
        step = max(1, (L - start) // max(1, sweep_points - 1))
        positions = sorted(set(list(range(start, L, step)) + [L]))
        for pos in positions:
            sweep_jobs.append((qid, pos))
    sweep_prompts = [USER_TEMPLATE.format(n=pos, prefix=annot[qid]["question"][:pos])
                     for qid, pos in sweep_jobs]
    return qids, sweep_jobs, sweep_prompts


def _eval_report(test_rows, annot, b_pred, b_labels, b_region, qids, sweep_jobs,
                 sweep_conf, threshold, subdir, split, title):
    """予測 conf（b_pred / sweep_conf）から A/B/C を計算・整形して dict を返す。"""
    import math
    import statistics

    abs_err = [abs(p - l) for p, l in zip(b_pred, b_labels)]
    mae_b = statistics.mean(abs_err)
    rmse_b = math.sqrt(statistics.mean([(p - l) ** 2 for p, l in zip(b_pred, b_labels)]))

    def pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return float("nan")
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    corr_b = pearson(b_pred, b_labels)
    in_mae = [e for e, rg in zip(abs_err, b_region) if rg]
    out_mae = [e for e, rg in zip(abs_err, b_region) if not rg]

    curves = {}
    for (qid, pos), c in zip(sweep_jobs, sweep_conf):
        curves.setdefault(qid, []).append((pos, c))
    c_full = [sorted(curves[qid])[-1][1] for qid in qids if curves.get(qid)]
    A = _buzz_metrics(qids, curves, annot, threshold)
    c_mean = statistics.mean(c_full) if c_full else float("nan")
    c_median = statistics.median(c_full) if c_full else float("nan")
    c_ge08 = sum(1 for c in c_full if c >= 0.8) / len(c_full) if c_full else float("nan")
    c_min = min(c_full) if c_full else float("nan")

    print(f"\n============ {title} ============")
    print(f"[A] buzz位置 MAE（移行ゲート MAE≤8文字・θ={threshold}）:")
    print(f"    questions={A['n']}  MAE={A['mae']:.2f}文字  median={A['median']:.2f}  "
          f"|err|≤8={A['within8']*100:.1f}%  閾値未交差={A['no_cross']}")
    print("[A'] θスイープ（オーケストレータ調整前提・最良閾値での到達精度）:")
    for th, mae, w8, nc in A["th_results"]:
        mark = "  ← best" if A["best"] and abs(th - A["best"][0]) < 1e-9 else ""
        print(f"    θ={th:.2f}  MAE={mae:5.2f}  ≤8={w8*100:5.1f}%  未交差={nc}{mark}")
    if A["best"]:
        print(f"    → 最良 θ={A['best'][0]:.2f} で MAE={A['best'][1]:.2f}文字  "
              f"（gate {'PASS' if A['best'][1] <= 8 else 'FAIL'}）")
    print(f"[B] confidence 回帰（{subdir}:{split}・n={len(test_rows)}）:")
    print(f"    MAE={mae_b:.4f}  RMSE={rmse_b:.4f}  Pearson r={corr_b:.4f}")
    if in_mae and out_mae:
        print(f"    buzz周辺 MAE={statistics.mean(in_mae):.4f}（n={len(in_mae)}） / "
              f"周辺外 MAE={statistics.mean(out_mae):.4f}（n={len(out_mae)}）")
    print("[C] 全文(100%) confidence:")
    print(f"    mean={c_mean:.3f}  median={c_median:.3f}  min={c_min:.3f}  ≥0.8={c_ge08*100:.1f}%")
    gate = "PASS" if (A["best"] and A["best"][1] <= 8) else "FAIL"
    print(f"[gate] 最良θでの MAE≤8文字 → {gate}")
    print("=" * (len(title) + 26) + "\n")

    return {
        "A": {"mae": A["mae"], "median": A["median"], "within8": A["within8"],
              "no_cross": A["no_cross"], "n": A["n"],
              "best_threshold": A["best"][0] if A["best"] else None,
              "best_mae": A["best"][1] if A["best"] else None},
        "B": {"mae": mae_b, "rmse": rmse_b, "pearson": corr_b,
              "in_mae": statistics.mean(in_mae) if in_mae else None,
              "out_mae": statistics.mean(out_mae) if out_mae else None},
        "C": {"mean": c_mean, "median": c_median, "min": c_min, "ge08": c_ge08},
    }


@app.function(
    image=image, gpu="A10G", cpu=4.0, timeout=60 * 60,
    volumes={"/hf": hf_cache}, secrets=secrets,
)
def evaluate(model_repo: str, dataset_repo: str, n_questions: int,
             threshold: float, conf_split: str, batch_size: int,
             max_seq_length: int, sweep_points: int):
    """回帰ヘッド版（prefix 素読み・生成なし）の評価。"""
    token = os.environ.get("HF_TOKEN")
    model, tok = _load_regressor(model_repo, max_seq_length, token)
    model.to("cuda")
    test_rows, annot, subdir, split = _load_eval_data(dataset_repo, conf_split, token)
    print(f"[eval] reg model={model_repo}  test={len(test_rows)} ({subdir}:{split})  "
          f"annotated={len(annot)}  threshold={threshold}")

    b_prompts = [r["messages"][0]["content"] for r in test_rows]
    b_labels = [float(r["meta"]["confidence_label"]) for r in test_rows]
    b_region = [bool(r["meta"].get("is_buzz_region")) for r in test_rows]
    b_pred = _predict_conf(model, tok, b_prompts, batch_size, max_seq_length)

    qids, sweep_jobs, sweep_prompts = _build_sweep(test_rows, annot, n_questions, sweep_points)
    sweep_conf = _predict_conf(model, tok, sweep_prompts, batch_size, max_seq_length)
    return _eval_report(test_rows, annot, b_pred, b_labels, b_region, qids, sweep_jobs,
                        sweep_conf, threshold, subdir, split, "buzz reg eval")


@app.function(
    image=image, gpu="A10G", cpu=4.0, timeout=60 * 60 * 2,
    volumes={"/hf": hf_cache}, secrets=secrets,
)
def evaluate_dual(model_repo: str, dataset_repo: str, n_questions: int,
                  threshold: float, conf_split: str, batch_size: int,
                  max_seq_length: int, sweep_points: int, max_think_tokens: int):
    """デュアルヘッド(A) の評価。各 prefix で <think> を生成し post-think hidden を回帰ヘッドで読む。
    生成を伴うため reg 版より遅い（A10G timeout 2h）。"""
    token = os.environ.get("HF_TOKEN")
    model, tok = _load_dual(model_repo, max_seq_length, token)
    model.to("cuda")
    test_rows, annot, subdir, split = _load_eval_data(dataset_repo, conf_split, token)
    print(f"[eval] dual model={model_repo}  test={len(test_rows)} ({subdir}:{split})  "
          f"annotated={len(annot)}  threshold={threshold}  think_tok={max_think_tokens}")

    b_prompts = [r["messages"][0]["content"] for r in test_rows]
    b_labels = [float(r["meta"]["confidence_label"]) for r in test_rows]
    b_region = [bool(r["meta"].get("is_buzz_region")) for r in test_rows]
    b_pred = _predict_conf_dual(model, tok, b_prompts, batch_size, max_seq_length,
                                max_think_tokens)

    qids, sweep_jobs, sweep_prompts = _build_sweep(test_rows, annot, n_questions, sweep_points)
    sweep_conf = _predict_conf_dual(model, tok, sweep_prompts, batch_size, max_seq_length,
                                    max_think_tokens)
    return _eval_report(test_rows, annot, b_pred, b_labels, b_region, qids, sweep_jobs,
                        sweep_conf, threshold, subdir, split, "buzz dual eval")


@app.local_entrypoint()
def train_main(base_model: str = "LiquidAI/LFM2.5-1.2B-JP-202606",
               data_subdir: str = "sft_corpus_1_steep",
               dataset_repo: str = "YUGOROU/quiz-ai-corpus",
               push_to_hub: str = "", public: bool = False,
               epochs: float = 2.0, batch_size: int = 256, grad_accum: int = 1,
               learning_rate: float = 1e-4, max_seq_length: int = 512,
               warmup_ratio: float = 0.05, max_steps: int = 0, seed: int = 3407,
               gpu: str = "", grad_ckpt: bool = False, spawn: bool = False):
    # メモリ非律速（bs64 でも peak ~9GiB / 40GB）。既定 grad_ckpt=False + 大バッチで
    # A100 を活かす。A10G(24GB)でも bs256 は収まる見込み（buzz は A10G 右サイズ）。
    # 疎通: modal run train/buzz_reg.py::train_main --max-steps 10 --batch-size 8
    # 本番: modal run train/buzz_reg.py::train_main \
    #         --push-to-hub YUGOROU/quiz-buzz-reg-1.2bjp --gpu A100-40GB
    fn = train.with_options(gpu=gpu) if gpu else train
    kw = dict(base_model=base_model, data_subdir=data_subdir, dataset_repo=dataset_repo,
              push_to_hub=(push_to_hub or None), hub_private=not public,
              epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
              learning_rate=learning_rate, max_seq_length=max_seq_length,
              warmup_ratio=warmup_ratio, max_steps=max_steps, seed=seed,
              grad_ckpt=grad_ckpt)
    if spawn:
        call = fn.spawn(**kw)
        print(f"[spawn] サーバー側で実行開始（Mac切断耐性）。call_id={call.object_id}")
        print("  進捗: modal app logs <app-id>。HFに push されたら完了。")
    else:
        fn.remote(**kw)


@app.local_entrypoint()
def eval_main(model_repo: str = "YUGOROU/quiz-buzz-reg-1.2bjp-merged",
              dataset_repo: str = "YUGOROU/quiz-ai-corpus",
              n_questions: int = 200, threshold: float = 0.5,
              conf_split: str = "sft_corpus_1_steep:test", batch_size: int = 64,
              max_seq_length: int = 512, sweep_points: int = 20):
    res = evaluate.remote(
        model_repo=model_repo, dataset_repo=dataset_repo, n_questions=n_questions,
        threshold=threshold, conf_split=conf_split, batch_size=batch_size,
        max_seq_length=max_seq_length, sweep_points=sweep_points,
    )
    print("result:", res)


@app.local_entrypoint()
def train_dual_main(base_model: str = "LiquidAI/LFM2.5-1.2B-JP-202606",
                    data_subdir: str = "sft_corpus_1_steep",
                    dataset_repo: str = "YUGOROU/quiz-ai-corpus",
                    push_to_hub: str = "", public: bool = False,
                    epochs: float = 2.0, batch_size: int = 64, grad_accum: int = 2,
                    learning_rate: float = 5e-5, max_seq_length: int = 512,
                    warmup_ratio: float = 0.1, max_steps: int = 0, seed: int = 3407,
                    reg_weight: float = 3.0, gpu: str = "", grad_ckpt: bool = True,
                    spawn: bool = False):
    # full-FT の CausalLM(CLM項)は LR に敏感（reg ヘッド単体の 1e-4 だと発散気味）。
    # LR=5e-5・warmup=0.1 で安定化。reg_weight=3 で BCE 項と CLM 項を釣り合わせる。
    # dual は全語彙ロジット[B,T,V]が乗りメモリ律速 → grad_ckpt=True・bs64(実効128)が既定。
    # デュアルヘッド(A): CLM(<think>生成) + 回帰ヘッド(post-think calibration) を同時学習。
    # 疎通: modal run train/buzz_reg.py::train_dual_main --max-steps 8 --batch-size 8
    # 本番: modal run train/buzz_reg.py::train_dual_main \
    #         --push-to-hub YUGOROU/quiz-buzz-dual-1.2bjp --gpu A100-40GB
    fn = train_dual.with_options(gpu=gpu) if gpu else train_dual
    kw = dict(base_model=base_model, data_subdir=data_subdir, dataset_repo=dataset_repo,
              push_to_hub=(push_to_hub or None), hub_private=not public,
              epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
              learning_rate=learning_rate, max_seq_length=max_seq_length,
              warmup_ratio=warmup_ratio, max_steps=max_steps, seed=seed,
              grad_ckpt=grad_ckpt, reg_weight=reg_weight)
    if spawn:
        call = fn.spawn(**kw)
        print(f"[spawn] サーバー側で実行開始（Mac切断耐性）。call_id={call.object_id}")
        print("  進捗: modal app logs <app-id>。HFに push されたら完了。")
    else:
        fn.remote(**kw)


@app.local_entrypoint()
def eval_dual_main(model_repo: str = "YUGOROU/quiz-buzz-dual-1.2bjp-merged",
                   dataset_repo: str = "YUGOROU/quiz-ai-corpus",
                   n_questions: int = 200, threshold: float = 0.5,
                   conf_split: str = "sft_corpus_1_steep:test", batch_size: int = 64,
                   max_seq_length: int = 512, sweep_points: int = 20,
                   max_think_tokens: int = 48):
    res = evaluate_dual.remote(
        model_repo=model_repo, dataset_repo=dataset_repo, n_questions=n_questions,
        threshold=threshold, conf_split=conf_split, batch_size=batch_size,
        max_seq_length=max_seq_length, sweep_points=sweep_points,
        max_think_tokens=max_think_tokens,
    )
    print("result:", res)
