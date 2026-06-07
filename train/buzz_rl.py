"""buzz 単独 RL（メイン凍結・GRPO-1 を buzz に限定）— precompute ＋ RL の自己完結 Modal。

背景（docs/HANDOFF.md・メモリ buzz-mae-not-capacity-bound / grpo-downgrade-grpo1-only）:
  buzz の SFT 残差は「S-buzz 疑似ラベルの buzz周辺ノイズ床」。SFT はラベルにフィットするので
  これ以上下がらない。RL は**ラベルでなく報酬を直接最適化**するのでノイズ床を迂回でき、かつ
  S-buzz アノテ(deepseek)と実デプロイ解答者(gemma-4 SFT)の**キャリブレーション・ギャップ**を
  実測で閉じられる。メインを凍結し buzz 方策のみ更新＝軽量（＝GRPO-1 warmup を buzz に限定）。

2段構成:
  1) precompute: 実デプロイ解答者（SFTメイン `quiz-main-gemma-merged`・4bit＝5090相当）が
     各問の「位置ごとに正答できるか」曲線をオフライン一括計算しルックアップ化（RLループに
     26B を入れない／同時にギャップを実測）。出力は Volume `quiz-corpus` の jsonl。
  2) RL（後日この同ファイルに追加）: 純回帰ヘッド buzz を初期方策に、conf=P(buzz) の逐次
     Bernoulli 方策を precompute 報酬で REINFORCE/GRPO 更新。

precompute 実行（パイロット）:
  uv run --with modal modal run train/buzz_rl.py::precompute_main \
    --main-repo YUGOROU/quiz-main-gemma-merged --split train --n-questions 300
"""
import os

import modal

app = modal.App("quiz-buzz-rl")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))   # qutils.is_correct を再利用

# gemma-4 ロード用（eval_knowledge.py と同一系統の依存）。
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch", "transformers", "accelerate", "bitsandbytes",
        "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
        "einops", "pillow", "tiktoken", "blobfile",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
    .add_local_dir(_SRC, "/opt/quizsrc", copy=True, ignore=["__pycache__", "*.pyc"])
)
SRC_DIR = "/opt/quizsrc"

corpus_vol = modal.Volume.from_name("quiz-corpus", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]

# RL 方策の buzz 位置サンプリングと同じ位置グリッド（fraction）。precompute はこの位置で
# メイン正答を測り、RL はこの同じ位置でしか buzz しない（報酬ルックアップと整合）。
POS_FRACTIONS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]


def _positions_for(L: int):
    """問題長 L に対する評価位置（昇順・10文字以上・重複/範囲外除去・末尾Lを含む）。"""
    ps = set()
    for f in POS_FRACTIONS:
        p = max(10, int(L * f))
        if p < L:
            ps.add(p)
    ps.add(L)
    return sorted(ps)


# gemma-4-26B-A4B は MoE で bnb 4bit がエキスパートを量子化しきれず bf16(~52GB)のまま残る
# → 40GB に収まらず CPU offload で落ちる。eval_knowledge.py と同じく H100(80GB) bf16 が確実。
@app.function(
    image=image, gpu="H100", cpu=8.0, timeout=60 * 60 * 4,
    volumes={"/data": corpus_vol, "/hf": hf_cache}, secrets=secrets,
)
def precompute(main_repo: str, split: str, n_questions: int, seed: int,
               batch_size: int, max_new_tokens: int, load_4bit: bool,
               out_name: str):
    """SFTメインの「位置ごと正答曲線」を一括計算して Volume に保存。"""
    import json
    import random
    import sys

    import torch

    sys.path.insert(0, SRC_DIR)
    import qutils

    token = os.environ.get("HF_TOKEN")

    # --- tokenizer / model（eval_knowledge.py と同経路・4bit はデプロイ(5090)相当） ---
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(main_repo, trust_remote_code=True, token=token)
    except Exception:  # noqa: BLE001
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(main_repo, trust_remote_code=True, token=token)
    quant_cfg = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    from transformers import AutoModelForCausalLM
    kw = dict(device_map="auto", trust_remote_code=True, token=token)
    if quant_cfg is not None:
        kw["quantization_config"] = quant_cfg
    else:
        kw["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(main_repo, **kw)
    model.eval()
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"     # バッチ生成は left padding

    eos_id = getattr(tok, "eos_token_id", None)

    def _tok2id(t):
        fn = getattr(tok, "convert_tokens_to_ids", None)
        try:
            tid = fn(t) if fn else None
        except Exception:  # noqa: BLE001
            tid = None
        unk = getattr(tok, "unk_token_id", None)
        return tid if (tid is not None and tid >= 0 and tid != unk) else None

    # gemma-4 のターン終端 <turn|>(id106) を stop に入れる（入れないと max_new_tokens まで
    # 反復生成＝コスト爆発。メモリ gemma4-turn-stop-token）。
    stop_ids = [i for i in dict.fromkeys([eos_id, _tok2id("<turn|>")]) if i is not None]
    print(f"[pre] main={main_repo} 4bit={load_4bit} stop_ids={stop_ids}")

    # --- 対象問題（annotated_questions.jsonl・qid 分割） ---
    rows = []
    with open("/data/annotated_questions.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("is_valid"):
                continue
            if split != "all" and qutils.qid_split(r["qid"]) != split:
                continue
            rows.append(r)
    random.Random(seed).shuffle(rows)
    rows = rows[:n_questions]
    print(f"[pre] split={split} questions={len(rows)}")

    # 全 (qid,pos) ジョブを1リスト化してバッチ生成
    jobs = []   # (qid, pos)
    for r in rows:
        for p in _positions_for(r["question_length"]):
            jobs.append((r["qid"], p))
    by_qid = {r["qid"]: r for r in rows}
    print(f"[pre] jobs={len(jobs)}  positions/q≈{len(jobs)/max(1,len(rows)):.1f}  "
          f"max_new_tokens={max_new_tokens}")

    @torch.no_grad()
    def answer_batch(user_contents):
        texts = []
        for u in user_contents:
            msgs = [{"role": "user", "content": u}]
            try:
                t = tok.apply_chat_template(msgs, enable_thinking=True,
                                            add_generation_prompt=True, tokenize=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            texts.append(t)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=eos_id, eos_token_id=stop_ids or eos_id)
        new = out[:, enc["input_ids"].shape[1]:]
        ans = []
        for row in new:
            g = tok.decode(row, skip_special_tokens=True)
            tail = g.split("</think>")[-1] if "</think>" in g else g
            tail = tail.strip()
            ans.append(tail.splitlines()[0].strip() if tail else "")
        return ans

    # バッチ実行 → curve[qid] = [(pos, correct, correct_loose)]
    curve = {}
    import time
    t0 = time.time()
    for i in range(0, len(jobs), batch_size):
        bj = jobs[i:i + batch_size]
        users = [
            f"早押しクイズ（{p}文字目時点）:\n{by_qid[qid]['question'][:p]}"
            for qid, p in bj
        ]
        preds = answer_batch(users)
        for (qid, p), pred in zip(bj, preds):
            golds = by_qid[qid]["answers"]
            c = int(qutils.is_correct(pred, golds))
            cl = int(qutils.is_correct(pred, golds, loose=True))
            curve.setdefault(qid, []).append((p, c, cl))
        if (i // batch_size) % 10 == 0:
            done = i + len(bj)
            rate = done / max(1e-9, time.time() - t0)
            print(f"[pre] {done}/{len(jobs)}  {rate:.1f} gen/s  "
                  f"eta {(len(jobs)-done)/max(1e-9,rate)/60:.1f}min")

    # --- 保存（Volume）。各 qid に gemma 正答曲線 ＋ 参照 buzz_char を付す ---
    out_path = f"/data/{out_name}"
    n_ans = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for qid, pts in curve.items():
            pts.sort()
            r = by_qid[qid]
            # gemma が初めて（loose）正答できる最小位置＝実機 can-answer 境界
            first = next((p for p, _, cl in pts if cl), None)
            n_ans += int(first is not None)
            w.write(json.dumps({
                "qid": qid,
                "question": r["question"],          # prefix 生成用（RL が self-contained に）
                "question_length": r["question_length"],
                "buzz_char_sbuzz": r["buzz_char"],
                "gemma_first_correct": first,
                "answers": r["answers"],
                "curve": [[p, c, cl] for p, c, cl in pts],
            }, ensure_ascii=False) + "\n")
    corpus_vol.commit()

    # サマリ: S-buzz と gemma can-answer 境界の乖離（キャリブレーション・ギャップの実測）
    import statistics
    gaps = []
    for qid, pts in curve.items():
        first = next((p for p, _, cl in pts if cl), None)
        if first is not None:
            gaps.append(first - by_qid[qid]["buzz_char"])
    print(f"[pre] saved {len(curve)} 問 -> {out_path}  "
          f"（gemma が答えられた問: {n_ans}/{len(curve)}）")
    if gaps:
        print(f"[pre] calib gap (gemma_first_correct - S-buzz): "
              f"mean={statistics.mean(gaps):+.1f}  median={statistics.median(gaps):+.1f}  "
              f"文字（正なら gemma は S-buzz より遅い位置でないと答えられない＝S-buzzが楽観的）")
    hf_cache.commit()
    return {"questions": len(curve), "answered": n_ans, "jobs": len(jobs),
            "calib_gap_mean": (statistics.mean(gaps) if gaps else None)}


@app.local_entrypoint()
def precompute_main(main_repo: str = "YUGOROU/quiz-main-gemma-merged",
                    split: str = "train", n_questions: int = 300, seed: int = 3407,
                    batch_size: int = 32, max_new_tokens: int = 192,
                    load_4bit: bool = False, out_name: str = "",
                    gpu: str = ""):
    # パイロット: modal run train/buzz_rl.py::precompute_main --n-questions 300
    # 本番subset:  modal run train/buzz_rl.py::precompute_main --n-questions 4000 --gpu H100
    name = out_name or f"gemma_cananswer_{split}_{n_questions}.jsonl"
    fn = precompute.with_options(gpu=gpu) if gpu else precompute
    res = fn.remote(main_repo=main_repo, split=split, n_questions=n_questions, seed=seed,
                    batch_size=batch_size, max_new_tokens=max_new_tokens,
                    load_4bit=load_4bit, out_name=name)
    print("result:", res, "out_name:", name)


# ============================================================
# RL（buzz 単独・グループREINFORCE＝GRPO-1 を buzz に限定）
#   方策 = 純回帰ヘッド buzz（conf=sigmoid=P(buzz)）。precompute の gemma 正答曲線を報酬源に、
#   逐次 Bernoulli 停止方策を group 正規化アドバンテージで REINFORCE。KL を初期方策へ掛け崩壊防止。
#   メインは凍結（precompute ルックアップのみ）＝ループに 26B を入れない。
# ============================================================
USER_TEMPLATE = "問題文（{n}文字目まで）:\n{prefix}"   # buzz_reg と完全一致
HEAD_FILE = "buzz_head.pt"


@app.function(
    image=image, gpu="A100-40GB", cpu=8.0, timeout=60 * 60 * 4,
    volumes={"/data": corpus_vol, "/hf": hf_cache}, secrets=secrets,
)
def rl_train(policy_repo: str, lookup_name: str, dataset_repo: str,
             push_to_hub: str | None, hub_private: bool,
             epochs: float, batch_questions: int, group_size: int, lr: float,
             max_seq_length: int, kl_coef: float, entropy_coef: float,
             reward_correct: float, reward_pos_bonus: float, reward_wrong: float,
             reward_skip: float, judge_loose: bool, grad_ckpt: bool, seed: int):
    import json
    import math
    import random
    import statistics

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    random.seed(seed)
    torch.manual_seed(seed)
    dev = "cuda"

    # --- 方策（純回帰ヘッド）をロード ---
    tok = AutoTokenizer.from_pretrained(policy_repo, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    backbone = AutoModel.from_pretrained(policy_repo, dtype="bfloat16", token=token).to(dev)
    if grad_ckpt:
        backbone.gradient_checkpointing_enable()
        backbone.config.use_cache = False
    H = backbone.config.hidden_size
    head = nn.Linear(H, 1).to(dev, dtype=next(backbone.parameters()).dtype)
    from huggingface_hub import hf_hub_download
    head.load_state_dict(torch.load(hf_hub_download(policy_repo, HEAD_FILE, token=token),
                                    map_location="cpu"))
    head.to(dev, dtype=next(backbone.parameters()).dtype)

    def confs_for(prompts):
        """prompts -> conf テンソル [N]（grad あり）。最終非padトークンプーリング。"""
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_seq_length).to(dev)
        hs = backbone(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"]).last_hidden_state
        last = enc["attention_mask"].long().sum(1) - 1
        pooled = hs[torch.arange(hs.size(0), device=dev), last]
        return torch.sigmoid(head(pooled).squeeze(-1).float())

    # --- 報酬源（gemma 正答曲線ルックアップ・Volume） ---
    lk = {}
    with open(f"/data/{lookup_name}") as f:
        for line in f:
            r = json.loads(line)
            ci = 2 if judge_loose else 1     # curve 各要素 = [pos, strict, loose]
            lk[r["qid"]] = {
                "L": r["question_length"],
                "sbuzz": r["buzz_char_sbuzz"],
                "positions": [c[0] for c in r["curve"]],
                "correct": {c[0]: c[ci] for c in r["curve"]},
            }
    qids = list(lk.keys())

    # 問題文（prefix 生成用）は lookup に無いので annotated_questions.jsonl（Volume）から qid 結合。
    # lookup に "question" があればそちらを優先（将来の自己完結 lookup 用）。
    questions = {}
    with open("/data/annotated_questions.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                questions[r["qid"]] = r["question"]
    qids = [q for q in qids if q in questions]

    def mkprompt(qid, p):
        return USER_TEMPLATE.format(n=p, prefix=questions[qid][:p])

    print(f"[rl] policy={policy_repo}  questions={len(qids)}  G={group_size} "
          f"bq={batch_questions} lr={lr} kl={kl_coef} judge={'loose' if judge_loose else 'strict'}")

    # --- 参照方策 conf（初期方策・固定）を一括 precompute（KL の基準・崩壊防止） ---
    ref_conf = {}
    backbone.eval()
    with torch.no_grad():
        flat = [(qid, p) for qid in qids for p in lk[qid]["positions"]]
        for i in range(0, len(flat), 256):
            chunk = flat[i:i + 256]
            cs = confs_for([mkprompt(qid, p) for qid, p in chunk])
            for (qid, p), c in zip(chunk, cs.tolist()):
                ref_conf[(qid, p)] = c
    print(f"[rl] ref conf precomputed: {len(ref_conf)} (qid,pos)")

    params = list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(qids) / batch_questions)
    total_steps = int(steps_per_epoch * epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    eps = 1e-6

    def reward_for(qid, buzz_idx):
        d = lk[qid]
        if buzz_idx is None:                       # スルー
            return reward_skip if any(d["correct"].values()) else 0.0
        pos = d["positions"][buzz_idx]
        if d["correct"].get(pos, 0):
            return reward_correct + reward_pos_bonus * (1.0 - pos / d["L"])
        return reward_wrong

    step = 0
    backbone.train()
    for ep in range(math.ceil(epochs)):
        random.shuffle(qids)
        for bi in range(0, len(qids), batch_questions):
            bq = qids[bi:bi + batch_questions]
            jobs = [(qid, p) for qid in bq for p in lk[qid]["positions"]]
            confs = confs_for([mkprompt(qid, p) for qid, p in jobs])
            # qid -> その位置順 conf テンソルのリスト
            idx = 0
            cmap = {}
            for qid in bq:
                n = len(lk[qid]["positions"])
                cmap[qid] = confs[idx:idx + n]
                idx += n

            loss = torch.zeros((), device=dev)
            ep_rewards, ep_buzzpos = [], []
            for qid in bq:
                cq = cmap[qid]                      # [n] grad
                n = cq.shape[0]
                # KL（current || ref）を全位置で（崩壊防止）
                rc = torch.tensor([ref_conf[(qid, p)] for p in lk[qid]["positions"]],
                                  device=dev).clamp(eps, 1 - eps)
                cqc = cq.clamp(eps, 1 - eps)
                kl = (cqc * (cqc / rc).log() + (1 - cqc) * ((1 - cqc) / (1 - rc)).log()).sum()
                ent = -(cqc * cqc.log() + (1 - cqc) * (1 - cqc).log()).sum()
                # G rollouts（同じ conf を共有）
                Rs, logps = [], []
                with torch.no_grad():
                    samp = (torch.rand(group_size, n, device=dev) < cq.detach()).int()
                for g in range(group_size):
                    buzz_idx, terms = None, []
                    for j in range(n):
                        if samp[g, j].item():
                            terms.append((cqc[j]).log())
                            buzz_idx = j
                            break
                        terms.append((1 - cqc[j]).log())
                    Rs.append(reward_for(qid, buzz_idx))
                    logps.append(torch.stack(terms).sum())
                    ep_buzzpos.append(lk[qid]["positions"][buzz_idx] if buzz_idx is not None
                                      else lk[qid]["L"])
                Rt = torch.tensor(Rs, device=dev)
                adv = (Rt - Rt.mean()) / (Rt.std() + 1e-6)
                pg = -(adv * torch.stack(logps)).mean()
                loss = loss + pg + kl_coef * kl - entropy_coef * ent
                ep_rewards.extend(Rs)
            loss = loss / len(bq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == 1:
                print(f"[rl] ep{ep} step{step}/{total_steps}  loss={loss.item():.4f}  "
                      f"meanR={statistics.mean(ep_rewards):+.3f}  "
                      f"meanBuzz={statistics.mean(ep_buzzpos):.1f}")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    # --- 保存（buzz_reg と同形式＝既存 eval がそのまま使える） ---
    out_dir = "/tmp/buzz_rl_merged"
    backbone.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(head.state_dict(), os.path.join(out_dir, HEAD_FILE))
    print(f"[rl] saved policy(backbone+{HEAD_FILE}) -> {out_dir}")
    if push_to_hub:
        repo = f"{push_to_hub}-merged"
        api = HfApi()
        api.create_repo(repo, private=hub_private, exist_ok=True, token=token)
        api.upload_folder(folder_path=out_dir, repo_id=repo, token=token)
        print(f"[rl] pushed ({'private' if hub_private else 'public'}) -> {repo}")
    corpus_vol.commit()
    hf_cache.commit()
    return {"questions": len(qids), "final_meanR": statistics.mean(ep_rewards)}


@app.function(
    image=image, gpu="A10G", cpu=4.0, timeout=60 * 60,
    volumes={"/data": corpus_vol, "/hf": hf_cache}, secrets=secrets,
)
def rl_eval(policy_repo: str, lookup_name: str, threshold: float,
            max_seq_length: int, judge_loose: bool, reward_correct: float,
            reward_pos_bonus: float, reward_wrong: float, reward_skip: float):
    """RL 方策の本来の成功基準を測る: 貪欲 buzz 位置 vs gemma 実 can-answer 境界の MAE と
    平均報酬。比較用に S-buzz buzz_char との MAE も出す（旧ゲート）。"""
    import json
    import statistics

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(policy_repo, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    backbone = AutoModel.from_pretrained(policy_repo, dtype="bfloat16", token=token).to(dev)
    backbone.eval()
    H = backbone.config.hidden_size
    head = nn.Linear(H, 1).to(dev, dtype=next(backbone.parameters()).dtype)
    head.load_state_dict(torch.load(hf_hub_download(policy_repo, HEAD_FILE, token=token),
                                    map_location="cpu"))
    head.to(dev, dtype=next(backbone.parameters()).dtype)

    lk = {}
    with open(f"/data/{lookup_name}") as f:
        for line in f:
            r = json.loads(line)
            ci = 2 if judge_loose else 1
            lk[r["qid"]] = {
                "q": r.get("question"), "L": r["question_length"],
                "sbuzz": r["buzz_char_sbuzz"], "first": r["gemma_first_correct"],
                "positions": [c[0] for c in r["curve"]],
                "correct": {c[0]: c[ci] for c in r["curve"]},
            }
    # question が lookup に無い旧形式は annotated から補完
    if any(v["q"] is None for v in lk.values()):
        with open("/data/annotated_questions.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r["qid"] in lk and lk[r["qid"]]["q"] is None:
                        lk[r["qid"]]["q"] = r["question"]
    qids = [q for q in lk if lk[q]["q"] is not None]

    @torch.no_grad()
    def conf_curve(qid):
        ps = lk[qid]["positions"]
        prompts = [USER_TEMPLATE.format(n=p, prefix=lk[qid]["q"][:p]) for p in ps]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_seq_length).to(dev)
        hs = backbone(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"]).last_hidden_state
        last = enc["attention_mask"].long().sum(1) - 1
        c = torch.sigmoid(head(hs[torch.arange(hs.size(0), device=dev), last]).squeeze(-1).float())
        return list(zip(ps, c.tolist()))

    def greedy_buzz(curve, th):
        for p, c in curve:
            if c >= th:
                return p
        return curve[-1][0]

    def reward_at(qid, p):
        d = lk[qid]
        if d["correct"].get(p, 0):
            return reward_correct + reward_pos_bonus * (1 - p / d["L"])
        return reward_wrong

    curves = {q: conf_curve(q) for q in qids}
    sweep = [0.30 + 0.05 * i for i in range(9)]
    print(f"[rleval] policy={policy_repo}  questions={len(qids)}  lookup={lookup_name}")
    best = None
    for th in sweep:
        e_gem, e_sb, rewards = [], [], []
        for q in qids:
            bp = greedy_buzz(curves[q], th)
            if lk[q]["first"] is not None:
                e_gem.append(abs(bp - lk[q]["first"]))
            e_sb.append(abs(bp - lk[q]["sbuzz"]))
            rewards.append(reward_at(q, bp))
        mae_g = statistics.mean(e_gem) if e_gem else float("nan")
        mae_s = statistics.mean(e_sb)
        mr = statistics.mean(rewards)
        w8 = sum(1 for x in e_gem if x <= 8) / len(e_gem) if e_gem else float("nan")
        print(f"    θ={th:.2f}  MAE_gemma={mae_g:5.2f}(≤8 {w8*100:4.1f}%)  "
              f"MAE_sbuzz={mae_s:5.2f}  meanR={mr:+.3f}")
        if best is None or mr > best[3]:
            best = (th, mae_g, mae_s, mr)
    print(f"  → 最良(報酬最大) θ={best[0]:.2f}: MAE_gemma={best[1]:.2f}  "
          f"MAE_sbuzz={best[2]:.2f}  meanR={best[3]:+.3f}")
    return {"best_th": best[0], "mae_gemma": best[1], "mae_sbuzz": best[2], "meanR": best[3]}


@app.local_entrypoint()
def rl_eval_main(policy_repo: str = "YUGOROU/quiz-buzz-rl-1.2bjp-merged",
                 lookup_name: str = "gemma_cananswer_val_150.jsonl",
                 threshold: float = 0.5, max_seq_length: int = 512,
                 judge_loose: bool = True, reward_correct: float = 1.0,
                 reward_pos_bonus: float = 0.5, reward_wrong: float = -1.5,
                 reward_skip: float = -0.1):
    res = rl_eval.remote(policy_repo=policy_repo, lookup_name=lookup_name,
                         threshold=threshold, max_seq_length=max_seq_length,
                         judge_loose=judge_loose, reward_correct=reward_correct,
                         reward_pos_bonus=reward_pos_bonus, reward_wrong=reward_wrong,
                         reward_skip=reward_skip)
    print("result:", res)


@app.local_entrypoint()
def rl_train_main(policy_repo: str = "YUGOROU/quiz-buzz-reg-1.2bjp-merged",
                  lookup_name: str = "gemma_cananswer_train_300.jsonl",
                  dataset_repo: str = "YUGOROU/quiz-ai-corpus",
                  push_to_hub: str = "", public: bool = False,
                  epochs: float = 3.0, batch_questions: int = 16, group_size: int = 8,
                  lr: float = 2e-6, max_seq_length: int = 512,
                  kl_coef: float = 0.05, entropy_coef: float = 0.0,
                  reward_correct: float = 1.0, reward_pos_bonus: float = 0.5,
                  reward_wrong: float = -1.5, reward_skip: float = -0.1,
                  judge_loose: bool = True, grad_ckpt: bool = True, seed: int = 3407,
                  gpu: str = ""):
    # パイロット: modal run train/buzz_rl.py::rl_train_main \
    #   --lookup-name gemma_cananswer_train_300.jsonl
    # 本番 + push: ... --push-to-hub YUGOROU/quiz-buzz-rl-1.2bjp
    # 報酬は quiz-ai.md（正解1.0+0.5pos / 誤答-1.5 / スルー-0.1）。ガードレール|機会損失|<|誤答|。
    fn = rl_train.with_options(gpu=gpu) if gpu else rl_train
    res = fn.remote(policy_repo=policy_repo, lookup_name=lookup_name, dataset_repo=dataset_repo,
                    push_to_hub=(push_to_hub or None), hub_private=not public,
                    epochs=epochs, batch_questions=batch_questions, group_size=group_size,
                    lr=lr, max_seq_length=max_seq_length, kl_coef=kl_coef,
                    entropy_coef=entropy_coef, reward_correct=reward_correct,
                    reward_pos_bonus=reward_pos_bonus, reward_wrong=reward_wrong,
                    reward_skip=reward_skip, judge_loose=judge_loose, grad_ckpt=grad_ckpt,
                    seed=seed)
    print("result:", res)
