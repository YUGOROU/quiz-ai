"""Phase 0b E2E バックエンド — ハッカソン ZeroGPU 形（1 GPU 窓 = 1 セッション）の Modal 実走。

背景（docs/HANDOFF.md 🎯ハッカソン節 / メモリ hackathon-build-small-deploy-zerogpu）:
  デプロイ先が HF Space ZeroGPU（ephemeral GPU・`@spaces.GPU(duration=120)` の窓割り当て）
  に変わったため、Vast 永続箱の vLLM serve 2 本（serve/）＋ asyncio 投機オーケストレータは
  本線で使わない。ZeroGPU では「1 回の GPU 窓で 2 モデルを一度ロードし、その窓内で複数問の
  早押しを実走」する形が素直。本スクリプトはそれを Modal の単一 GPU 関数で先取り検証する
  （Modal も ZeroGPU も非永続＝本番近似。Modal で通れば `@spaces.GPU` にほぼ素移植できる）。

このスクリプトが測るもの（ZeroGPU 移植の判断材料）:
  1. 窓頭ロード時間（gemma メイン + buzz 回帰ヘッドを GPU に載せるまで）。120s 窓のうち
     何秒が推論に残るか。重ければ gemma 4bit(--load-4bit) に落とす判断。
  2. 1 問あたり実時間（buzz スキャン + gemma 生成）→ 1 窓に何問入るか。
  3. realized buzz 位置 vs S-buzz(buzz_char) と、その打点での answer 正解率（速度↔精度）。
     θ を上げる（打点を遅らせる）と正解率が上がる（HANDOFF θトレードオフ）。
  4. 毎 char buzz レイテンシ（≤300ms 予算）と buzz→answer レイテンシ（≤1s 予算）。

エピソード（1 問）:
  char-stream で prefix を伸ばし、各位置で buzz 回帰ヘッドの conf=sigmoid を見る。
  conf≥θ で初めて buzz → その prefix を corpus-2 と同一書式でメイン gemma に渡し
  `<think>…</think>answer` を生成 → `</think>` 以降を answer 抽出 → qutils.is_correct 採点。
  ※ ZeroGPU はローカルモデルで torch.generate を途中キャンセルできないため、投機推論
    （buzz 前にメイン先行 → no-buzz でキャンセル）は採らず「buzz 確定 → 生成」の同期形。
    投機の代わりに buzz→answer レイテンシを実測し、必要なら think budget で詰める。

モデル（再訓練なし・そのまま）:
  メイン = YUGOROU/quiz-main-gemma-merged（gemma-4-26B-A4B SFT・stop に <turn|>=106 必須）
  buzz   = YUGOROU/quiz-buzz-reg-1.2bjp-merged（回帰ヘッド・conf≥θ）

データ: Volume quiz-corpus の annotated_questions.jsonl（val split・buzz_char 参照付き）。

実行（スモーク・数問）:
  uv run --with modal modal run train/e2e_modal.py --n 5
本走（θ=0.55・打点を S-buzz より遅らせて精度寄り）:
  uv run --with modal modal run train/e2e_modal.py --n 50 --theta 0.55 --gpu H100
gemma 4bit（窓頭ロード短縮の検証）:
  uv run --with modal modal run train/e2e_modal.py --n 50 --load-4bit
"""
import os
import time

import modal

app = modal.App("quiz-e2e")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))  # qutils.is_correct を再利用

# eval_knowledge.py と同系の image（新アーキ gemma-4 を trust_remote_code でロード）。
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch", "transformers", "accelerate", "huggingface_hub", "hf_transfer",
        "sentencepiece", "protobuf", "einops", "pillow", "tiktoken", "blobfile",
        "triton", "bitsandbytes",
        # `kernels` は入れない（最新 transformers と版衝突・eval_knowledge.py 参照）。
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
    .add_local_dir(_SRC, "/opt/quizsrc", copy=True, ignore=["__pycache__", "*.pyc"])
)

SRC_DIR = "/opt/quizsrc"

corpus_vol = modal.Volume.from_name("quiz-corpus", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]

# 学習時（corpus-1 / serve/serve_buzz.py）と厳密一致させること。ズレると conf がずれる。
BUZZ_USER_TEMPLATE = "問題文（{n}文字目まで）:\n{prefix}"
HEAD_FILE = "buzz_head.pt"
# メイン gemma の入力は corpus-2 と同一書式（{n}文字目時点＝buzz した prefix 長）。
MAIN_USER_TEMPLATE = "早押しクイズ（{n}文字目時点）:\n{prefix}"


# ============================================================
# モデルロード（buzz 回帰ヘッド / メイン gemma）。torch 依存は GPU 関数内でのみ評価。
# ============================================================
def _load_buzz(repo: str, max_seq_length: int, token, device="cuda"):
    """serve/serve_buzz.py と同一: AutoModel backbone + 最終非padトークン pooling + Linear(h,1)。"""
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    class BuzzRegressor(nn.Module):
        def __init__(self, backbone, hidden_size):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(hidden_size, 1)
            self.head.to(dtype=next(backbone.parameters()).dtype)

        @torch.no_grad()
        def forward(self, input_ids=None, attention_mask=None, **_):
            hs = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
            last = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)  # 右パディング前提
            pooled = hs[torch.arange(hs.size(0), device=hs.device), last]
            return self.head(pooled).squeeze(-1)

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


def _load_main(repo: str, token, load_4bit: bool):
    """eval_knowledge.py の load 経路（auto-class カスケード + <turn|> stop）を踏襲。"""
    import torch
    from transformers import AutoTokenizer

    quant_cfg = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, token=token)
    except Exception:  # noqa: BLE001 — マルチモーダルは processor のことがある
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(repo, trust_remote_code=True, token=token)

    from transformers import AutoModelForCausalLM
    cands = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        cands.append(AutoModelForImageTextToText)
    except Exception:  # noqa: BLE001
        pass
    model, last = None, None
    for cls in cands:
        try:
            kw = dict(torch_dtype="auto", device_map="auto",
                      trust_remote_code=True, token=token)
            if quant_cfg is not None:
                kw["quantization_config"] = quant_cfg
                kw.pop("torch_dtype")
            model = cls.from_pretrained(repo, **kw)
            print(f"[load] main OK via {cls.__name__}  4bit={load_4bit}")
            break
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[load] {cls.__name__} 失敗: {type(e).__name__}: {str(e)[:160]}")
    if model is None:
        raise last
    model.eval()

    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is None and hasattr(tok, "tokenizer"):
        eos_id = getattr(tok.tokenizer, "eos_token_id", None)

    def _tok2id(t):
        fn = getattr(tok, "convert_tokens_to_ids", None)
        if fn is None and hasattr(tok, "tokenizer"):
            fn = getattr(tok.tokenizer, "convert_tokens_to_ids", None)
        try:
            tid = fn(t) if fn else None
        except Exception:  # noqa: BLE001
            tid = None
        unk = getattr(tok, "unk_token_id", None)
        return tid if (tid is not None and tid >= 0 and tid != unk) else None

    # gemma-4 のターン終端 <turn|>(id 106) を stop に必ず含める（漏れると反復生成でレイテンシ悪化）。
    stop_ids = [i for i in dict.fromkeys([eos_id, _tok2id("<turn|>")]) if i is not None]
    print(f"[stop] main eos_token_id={stop_ids} (<turn|>={_tok2id('<turn|>')})")
    return model, tok, eos_id, stop_ids


# ============================================================
# E2E 1 セッション = 1 GPU 窓（gemma + buzz を一度ロードし複数問を実走）
# ============================================================
@app.function(
    image=image, gpu="H100", cpu=4.0, timeout=60 * 60,
    volumes={"/data": corpus_vol, "/hf": hf_cache}, secrets=secrets,
)
def run_session(main_repo: str, buzz_repo: str, n_questions: int, theta: float,
                seed: int, stride: int, max_new_tokens: int, buzz_batch: int,
                max_seq_length: int, load_4bit: bool, window_seconds: float,
                think: bool):
    import json
    import random
    import statistics
    import sys

    import torch

    sys.path.insert(0, SRC_DIR)
    import qutils

    token = os.environ.get("HF_TOKEN")

    # --- 窓頭ロード（ZeroGPU 窓の頭で 2 モデルを GPU に載せる時間を計る） ---
    t_load0 = time.perf_counter()
    buzz, buzz_tok = _load_buzz(buzz_repo, max_seq_length, token)
    t_buzz_load = time.perf_counter() - t_load0
    main, main_tok, eos_id, stop_ids = _load_main(main_repo, token, load_4bit)
    t_load = time.perf_counter() - t_load0
    print(f"[load] buzz={t_buzz_load:.1f}s  +main={t_load - t_buzz_load:.1f}s  "
          f"total={t_load:.1f}s（120s 窓の残り≈{max(0.0, window_seconds - t_load):.0f}s）")

    # --- データ: annotated val split（buzz_char 参照付き・is_valid のみ） ---
    targets = []
    with open("/data/annotated_questions.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("is_valid"):
                continue
            if qutils.qid_split(r["qid"]) != "val":
                continue
            targets.append(r)
    random.Random(seed).shuffle(targets)
    targets = targets[:n_questions]
    print(f"[data] val 評価 {len(targets)} 問  θ={theta}  stride={stride}  "
          f"max_new_tokens={max_new_tokens}  think={think}")

    # --- buzz: prefix 群 → conf=sigmoid（バッチ forward） ---
    @torch.no_grad()
    def buzz_conf(texts):
        confs = []
        for i in range(0, len(texts), buzz_batch):
            b = texts[i:i + buzz_batch]
            enc = buzz_tok(b, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_seq_length).to("cuda")
            logits = buzz(**enc)
            confs.extend(torch.sigmoid(logits.float()).reshape(-1).tolist())
        return confs

    # --- 毎 char buzz レイテンシ（bs=1 単発 forward・≤300ms 予算） ---
    warm = BUZZ_USER_TEMPLATE.format(n=10, prefix=targets[0]["question"][:10])
    buzz_conf([warm])  # warmup
    t = time.perf_counter()
    for _ in range(20):
        buzz_conf([warm])
    buzz_lat_ms = (time.perf_counter() - t) / 20 * 1000
    print(f"[lat] buzz 単発(bs1) p̄={buzz_lat_ms:.1f}ms/char（≤300ms 予算）")

    # --- メイン: prefix → answer（corpus-2 書式・<think>…</think> 後を answer 抽出） ---
    @torch.no_grad()
    def main_answer(prefix: str):
        user = MAIN_USER_TEMPLATE.format(n=len(prefix), prefix=prefix)
        msgs = [{"role": "user", "content": user}]
        kw = dict(add_generation_prompt=True, tokenize=True,
                  return_dict=True, return_tensors="pt")
        try:
            inputs = main_tok.apply_chat_template(msgs, enable_thinking=think, **kw)
        except TypeError:
            inputs = main_tok.apply_chat_template(msgs, **kw)
        inputs = inputs.to(main.device)
        in_len = inputs["input_ids"].shape[1]
        out = main.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=eos_id, eos_token_id=stop_ids or eos_id)
        gen = main_tok.decode(out[0][in_len:], skip_special_tokens=True)
        tail = gen.split("</think>")[-1] if "</think>" in gen else gen
        ans = tail.strip().splitlines()[0].strip() if tail.strip() else ""
        return ans, gen

    # --- エピソードループ（窓内で複数問・累積時間を追い 1 窓何問かを推定） ---
    rows = []
    sess_t0 = time.perf_counter()
    for i, r in enumerate(targets, 1):
        q, golds = r["question"], r["answers"]
        L = r["question_length"]
        buzz_char = r["buzz_char"]

        # char-stream を stride で走査し conf≥θ の初出位置を buzz_pos とする。
        start = max(10, int(0.15 * L))
        positions = list(range(start, L + 1, max(1, stride)))
        if positions[-1] != L:
            positions.append(L)
        ts = time.perf_counter()
        confs = buzz_conf([BUZZ_USER_TEMPLATE.format(n=p, prefix=q[:p]) for p in positions])
        buzz_scan_ms = (time.perf_counter() - ts) * 1000

        buzz_pos, crossed = L, False
        for p, c in zip(positions, confs):
            if c >= theta:
                buzz_pos, crossed = p, True
                break

        # buzz した prefix でメイン推論。
        tg = time.perf_counter()
        pred, raw = main_answer(q[:buzz_pos])
        gen_s = time.perf_counter() - tg

        c_strict = qutils.is_correct(pred, golds)
        c_loose = qutils.is_correct(pred, golds, loose=True)
        rows.append(dict(
            qid=r["qid"], L=L, buzz_char=buzz_char, buzz_pos=buzz_pos,
            crossed=crossed, strict=c_strict, loose=c_loose,
            buzz_scan_ms=buzz_scan_ms, gen_s=gen_s))
        mark = "✅" if c_strict else ("➕" if c_loose else "❌")
        print(f"[{i}/{len(targets)}] {mark} buzz@{buzz_pos}/{L}"
              f"(S-buzz {buzz_char}, Δ{buzz_pos - buzz_char:+d}) "
              f"gen={gen_s:.2f}s gold={golds} pred={pred!r}"
              + ("" if crossed else "  [θ未交差→全文]"))

    # --- 集約 ---
    N = len(rows) or 1
    acc_s = sum(x["strict"] for x in rows) / N
    acc_l = sum(x["loose"] for x in rows) / N
    mean_buzz = statistics.mean(x["buzz_pos"] for x in rows)
    mean_sbuzz = statistics.mean(x["buzz_char"] for x in rows)
    mean_delta = statistics.mean(x["buzz_pos"] - x["buzz_char"] for x in rows)
    mean_ratio = statistics.mean(x["buzz_pos"] / x["L"] for x in rows)
    no_cross = sum(1 for x in rows if not x["crossed"])
    mean_gen = statistics.mean(x["gen_s"] for x in rows)
    mean_scan = statistics.mean(x["buzz_scan_ms"] for x in rows)
    mean_perq = statistics.mean(x["buzz_scan_ms"] / 1000 + x["gen_s"] for x in rows)
    sess_total = time.perf_counter() - sess_t0
    win_budget = max(0.0, window_seconds - t_load)
    q_per_window = (win_budget / mean_perq) if mean_perq > 0 else float("nan")

    print(f"\n[E2E] === main={main_repo} buzz={buzz_repo} θ={theta} 4bit={load_4bit} ===")
    print(f"  正解率（realized buzz 打点）: strict {acc_s:.1%} / loose {acc_l:.1%}"
          f"  → メイン側ゲート≥65%(loose): {'PASS' if acc_l >= 0.65 else 'FAIL'}")
    print(f"  打点: buzz̄={mean_buzz:.1f}字  S-buzz̄={mean_sbuzz:.1f}字  "
          f"Δ̄={mean_delta:+.1f}字  読了率̄={mean_ratio:.1%}  θ未交差={no_cross}/{N}")
    print(f"  レイテンシ: buzz単発={buzz_lat_ms:.0f}ms/char  buzzスキャン̄={mean_scan:.0f}ms/問  "
          f"gen̄={mean_gen:.2f}s/問（=buzz→answer・≤1s 予算）")
    print(f"  窓: 窓頭ロード={t_load:.1f}s  1問̄={mean_perq:.2f}s  "
          f"→ {window_seconds:.0f}s 窓に約 {q_per_window:.1f} 問（ロード後残{win_budget:.0f}s）")
    print(f"  セッション実時間（{N}問）={sess_total:.1f}s")
    return dict(n=N, strict=acc_s, loose=acc_l, mean_buzz=mean_buzz,
                mean_sbuzz=mean_sbuzz, mean_delta=mean_delta, no_cross=no_cross,
                load_s=t_load, mean_gen_s=mean_gen, mean_perq_s=mean_perq,
                buzz_lat_ms=buzz_lat_ms, q_per_window=q_per_window)


# ============================================================
# vLLM 版（メイン高速化・thinking ごと ≤1s を狙う）
#   transformers generate(~37tok/s) が ≤1s 予算を破る真因 → gemma を vLLM に載せる
#   （paged-attn + 最適化カーネルで 100–200 tok/s 級）。buzz は回帰ヘッドなので vLLM 不可
#   → transformers のまま同一 GPU 同居（H100 80GB: vLLM 先に確保 → buzz を残りに載せる）。
# ============================================================
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")  # CUDA graph/compile 用に gcc（無いと inductor が落ちる）
    # deep_gemm は FP8 MoE カーネル用（pip 単体ビルド不可・HANDOFF）。我々は bf16 なので不要。
    .uv_pip_install("vllm", "transformers", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",  # flashinfer JIT が nvcc 要求で落ちる回避
          # gemma-4 MoE で vLLM が FP8 deep_gemm カーネルのウォームアップを強制し、
          # deep_gemm 不在で EngineCore init が落ちる。重みは bf16 なので FP8 経路を無効化。
          "VLLM_USE_DEEP_GEMM": "0"})
    .add_local_dir(_SRC, "/opt/quizsrc", copy=True, ignore=["__pycache__", "*.pyc"])
)


@app.function(
    image=vllm_image, gpu="H100", cpu=4.0, timeout=60 * 60,
    volumes={"/data": corpus_vol, "/hf": hf_cache}, secrets=secrets,
)
def run_session_vllm(main_repo: str, buzz_repo: str, n_questions: int, theta: float,
                     seed: int, stride: int, max_new_tokens: int, buzz_batch: int,
                     max_seq_length: int, gpu_frac: float, window_seconds: float,
                     think: bool, enforce_eager: bool):
    import json
    import random
    import statistics
    import sys

    import torch

    sys.path.insert(0, SRC_DIR)
    import qutils

    token = os.environ.get("HF_TOKEN")

    # --- 窓頭ロード: vLLM(gemma) を先に確保 → buzz を残りに載せる ---
    t_load0 = time.perf_counter()
    from vllm import LLM, SamplingParams
    import vllm
    print(f"[vllm] version={vllm.__version__}  loading {main_repo} ...")
    llm = LLM(model=main_repo, dtype="bfloat16", max_model_len=max_seq_length * 2,
              gpu_memory_utilization=gpu_frac, enforce_eager=enforce_eager,
              trust_remote_code=True)
    t_main_load = time.perf_counter() - t_load0
    buzz, buzz_tok = _load_buzz(buzz_repo, max_seq_length, token)
    t_load = time.perf_counter() - t_load0
    print(f"[load] main(vLLM)={t_main_load:.1f}s  +buzz={t_load - t_main_load:.1f}s  "
          f"total={t_load:.1f}s（{window_seconds:.0f}s 窓の残り≈{max(0.0, window_seconds - t_load):.0f}s）")

    # gemma-4 のターン終端 <turn|>(106) を stop に必ず含める（serve_main.sh と同一）。
    # chat テンプレート適用は vLLM の llm.chat() が内部で行うため main 側 tokenizer は不要。
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=[1, 106])

    # --- データ: annotated val split ---
    targets = []
    with open("/data/annotated_questions.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("is_valid") and qutils.qid_split(r["qid"]) == "val":
                targets.append(r)
    random.Random(seed).shuffle(targets)
    targets = targets[:n_questions]
    print(f"[data] val 評価 {len(targets)} 問  θ={theta}  stride={stride}  "
          f"max_new_tokens={max_new_tokens}  think={think}  eager={enforce_eager}")

    @torch.no_grad()
    def buzz_conf(texts):
        confs = []
        for i in range(0, len(texts), buzz_batch):
            b = texts[i:i + buzz_batch]
            enc = buzz_tok(b, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_seq_length).to("cuda")
            logits = buzz(**enc)
            confs.extend(torch.sigmoid(logits.float()).reshape(-1).tolist())
        return confs

    def main_answer(prefix: str):
        """corpus-2 書式の user を vLLM chat に渡し `<think>…</think>answer` を生成。
        戻り (answer, 生出力, 生成トークン数)。"""
        user = MAIN_USER_TEMPLATE.format(n=len(prefix), prefix=prefix)
        msgs = [{"role": "user", "content": user}]
        out = llm.chat(msgs, sp, use_tqdm=False,
                       chat_template_kwargs={"enable_thinking": think})
        o = out[0].outputs[0]
        gen = o.text
        ntok = len(o.token_ids)
        tail = gen.split("</think>")[-1] if "</think>" in gen else gen
        ans = tail.strip().splitlines()[0].strip() if tail.strip() else ""
        return ans, gen, ntok

    # warmup（CUDA graph/コンパイル後の定常レイテンシを測るため1回空打ち）
    _ = main_answer(targets[0]["question"][:targets[0]["buzz_char"]])

    rows = []
    sess_t0 = time.perf_counter()
    for i, r in enumerate(targets, 1):
        q, golds = r["question"], r["answers"]
        L, buzz_char = r["question_length"], r["buzz_char"]
        start = max(10, int(0.15 * L))
        positions = list(range(start, L + 1, max(1, stride)))
        if positions[-1] != L:
            positions.append(L)
        confs = buzz_conf([BUZZ_USER_TEMPLATE.format(n=p, prefix=q[:p]) for p in positions])
        buzz_pos, crossed = L, False
        for p, c in zip(positions, confs):
            if c >= theta:
                buzz_pos, crossed = p, True
                break
        tg = time.perf_counter()
        pred, raw, ntok = main_answer(q[:buzz_pos])
        gen_s = time.perf_counter() - tg
        cs = qutils.is_correct(pred, golds)
        cl = qutils.is_correct(pred, golds, loose=True)
        rows.append(dict(qid=r["qid"], L=L, buzz_char=buzz_char, buzz_pos=buzz_pos,
                         crossed=crossed, strict=cs, loose=cl, gen_s=gen_s, ntok=ntok))
        mark = "✅" if cs else ("➕" if cl else "❌")
        print(f"[{i}/{len(targets)}] {mark} buzz@{buzz_pos}/{L}(S {buzz_char},Δ{buzz_pos-buzz_char:+d}) "
              f"gen={gen_s:.2f}s({ntok}tok,{ntok/max(gen_s,1e-9):.0f}t/s) gold={golds} pred={pred!r}")

    N = len(rows) or 1
    acc_s = sum(x["strict"] for x in rows) / N
    acc_l = sum(x["loose"] for x in rows) / N
    mean_gen = statistics.mean(x["gen_s"] for x in rows)
    mean_tok = statistics.mean(x["ntok"] for x in rows)
    tps = mean_tok / mean_gen if mean_gen else float("nan")
    mean_ratio = statistics.mean(x["buzz_pos"] / x["L"] for x in rows)
    mean_delta = statistics.mean(x["buzz_pos"] - x["buzz_char"] for x in rows)
    le1s = sum(1 for x in rows if x["gen_s"] <= 1.0) / N
    sess_total = time.perf_counter() - sess_t0

    print(f"\n[E2E-vLLM] === main={main_repo} θ={theta} think={think} eager={enforce_eager} ===")
    print(f"  正解率（buzz 打点）: strict {acc_s:.1%} / loose {acc_l:.1%}"
          f"  → ゲート≥65%(loose): {'PASS' if acc_l >= 0.65 else 'FAIL'}")
    print(f"  打点: Δ̄={mean_delta:+.1f}字  読了率̄={mean_ratio:.1%}")
    print(f"  レイテンシ: gen̄={mean_gen:.2f}s/問  {tps:.0f}tok/s  生成̄={mean_tok:.0f}tok  "
          f"≤1s 達成率={le1s:.0%}（buzz→answer ≤1s 予算）")
    print(f"  窓: 窓頭ロード={t_load:.1f}s  セッション実時間({N}問)={sess_total:.1f}s")
    return dict(n=N, strict=acc_s, loose=acc_l, mean_gen_s=mean_gen, tps=tps,
                mean_tok=mean_tok, le1s=le1s, load_s=t_load, mean_delta=mean_delta)


@app.local_entrypoint()
def vllm_run(main_repo: str = "YUGOROU/quiz-main-gemma-merged",
             buzz_repo: str = "YUGOROU/quiz-buzz-reg-1.2bjp-merged",
             n: int = 20, theta: float = 0.6, seed: int = 3407, stride: int = 2,
             max_new_tokens: int = 256, buzz_batch: int = 64, max_seq_length: int = 512,
             gpu_frac: float = 0.80, gpu: str = "", window_seconds: float = 120.0,
             think: bool = True, enforce_eager: bool = True):
    # vLLM 版 E2E。thinking ON のまま decode を速くして ≤1s を狙う。
    # スモーク:  modal run train/e2e_modal.py::vllm_run --n 8
    # CUDA graph で更に高速化:  --no-enforce-eager（gcc 必要・image に build-essential 済）
    fn = run_session_vllm.with_options(gpu=gpu) if gpu else run_session_vllm
    res = fn.remote(
        main_repo=main_repo, buzz_repo=buzz_repo, n_questions=n, theta=theta, seed=seed,
        stride=stride, max_new_tokens=max_new_tokens, buzz_batch=buzz_batch,
        max_seq_length=max_seq_length, gpu_frac=gpu_frac, window_seconds=window_seconds,
        think=think, enforce_eager=enforce_eager)
    print("result:", res)


@app.local_entrypoint()
def main(main_repo: str = "YUGOROU/quiz-main-gemma-merged",
         buzz_repo: str = "YUGOROU/quiz-buzz-reg-1.2bjp-merged",
         n: int = 5, theta: float = 0.55, seed: int = 3407, stride: int = 2,
         max_new_tokens: int = 256, buzz_batch: int = 64, max_seq_length: int = 512,
         load_4bit: bool = False, gpu: str = "", window_seconds: float = 120.0,
         think: bool = True):
    # θ=0.55 は HANDOFF の θトレードオフで打点を S-buzz より +数字 遅らせ精度寄りにする既定。
    # 純回帰ヘッドの best θ=0.45（MAE最小）より高め＝メイン側ゲートを通す運用点。
    # スモーク: modal run train/e2e_modal.py --n 5
    # 本走:     modal run train/e2e_modal.py --n 50 --theta 0.55 --gpu H100
    fn = run_session.with_options(gpu=gpu) if gpu else run_session
    res = fn.remote(
        main_repo=main_repo, buzz_repo=buzz_repo, n_questions=n, theta=theta,
        seed=seed, stride=stride, max_new_tokens=max_new_tokens, buzz_batch=buzz_batch,
        max_seq_length=max_seq_length, load_4bit=load_4bit, window_seconds=window_seconds,
        think=think)
    print("result:", res)
