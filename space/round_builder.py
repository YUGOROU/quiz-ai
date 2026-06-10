"""HF Space バックエンドの核 — 実モデルで `QUIZ_ROUND`（フロントエンド契約）を生成する。

統合契約（docs/HANDOFF.md「統合契約 precompute-round」）:
  Claude Design の big-screen エンジンは `window.QUIZ_ROUND` を受けてタイマー再生する
  スクリプト型。各問は {id, full, truth, buzzFrac, buzzer, answer, correct, aiThink:[{frac,text}]}。
  本モジュールは **ハードコードの buzzFrac/answer/aiThink を実モデルで置換**する:
    - buzzFrac : buzz 回帰ヘッド（conf≥θ の初出位置 / 問題長）
    - answer   : gemma-4 SFT（buzz した prefix から `<think>…</think>answer` 生成）
    - aiThink  : gemma の `<think>` を文分割し frac を ~0.2〜buzzFrac に配る（reading 中に
                 思考が立ち上がる UI を再現・デモライセンス）
    - correct  : qutils.is_correct（loose）で truth と照合

設計（docs/HANDOFF.md「レイテンシ方針確定」）:
  - thinking 必須（no-think は精度崩壊）。decode は数秒だが **1 GPU 窓で N問を precompute** し
    「マッチ読込」に吸収する（frontend は READ_CPS で滑らか再生・問題間待ちゼロ）。
  - ZeroGPU は vLLM 不要・transformers で十分（vLLM と同精度）。gemma stop に <turn|>(106) 必須。
  - buzz は回帰ヘッド（9ms/char）。同一プロセスに gemma(transformers)＋buzz を同居。

このモジュールは spaces/Modal 非依存（torch/transformers のみ）。`@spaces.GPU` 関数からも
Modal のテストハーネスからも import して使う。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

# 学習時（corpus-1 / serve/serve_buzz.py）と厳密一致。ズレると conf がずれる。
BUZZ_USER_TEMPLATE = "問題文（{n}文字目まで）:\n{prefix}"
HEAD_FILE = "buzz_head.pt"
# メイン gemma の入力は corpus-2 と同一書式（{n}文字目時点＝buzz した prefix 長）。
MAIN_USER_TEMPLATE = "早押しクイズ（{n}文字目時点）:\n{prefix}"

DEFAULT_MAIN_REPO = "YUGOROU/quiz-main-gemma-merged"
DEFAULT_BUZZ_REPO = "YUGOROU/quiz-buzz-reg-1.2bjp-merged"


@dataclass
class Models:
    main_model: object
    main_tok: object
    main_eos: int
    main_stop: list
    buzz_model: object
    buzz_tok: object
    device: str
    tts: object = None      # Irodori InferenceRuntime（問題読み上げ・無ければ音声なし）


# ============================================================
# ロード
# ============================================================
def _load_buzz(repo: str, max_seq_length: int, token, device):
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
            last = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)
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


def _load_main(repo: str, token, load_4bit: bool, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_cfg = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, token=token)
    except Exception:  # noqa: BLE001
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(repo, trust_remote_code=True, token=token)

    cands = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        cands.append(AutoModelForImageTextToText)
    except Exception:  # noqa: BLE001
        pass
    model, last = None, None
    for cls in cands:
        try:
            kw = dict(torch_dtype="auto", device_map=device,
                      trust_remote_code=True, token=token)
            if quant_cfg is not None:
                kw["quantization_config"] = quant_cfg
                kw.pop("torch_dtype")
            model = cls.from_pretrained(repo, **kw)
            break
        except Exception as e:  # noqa: BLE001
            last = e
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

    # gemma-4 のターン終端 <turn|>(106) を stop に必ず含める（漏れると反復生成）。
    stop_ids = [i for i in dict.fromkeys([eos_id, _tok2id("<turn|>")]) if i is not None]
    return model, tok, eos_id, stop_ids


def load_models(main_repo: str = DEFAULT_MAIN_REPO, buzz_repo: str = DEFAULT_BUZZ_REPO,
                token: str | None = None, device: str = "cuda",
                max_seq_length: int = 512, load_4bit: bool = False,
                load_tts: bool = False) -> Models:
    """gemma メイン + buzz 回帰ヘッド（+任意で Irodori TTS）を同一デバイスにロード。"""
    token = token or os.environ.get("HF_TOKEN")
    buzz_model, buzz_tok = _load_buzz(buzz_repo, max_seq_length, token, device)
    main_model, main_tok, eos, stop = _load_main(main_repo, token, load_4bit, device)
    tts = _load_tts(device) if load_tts else None
    return Models(main_model=main_model, main_tok=main_tok, main_eos=eos, main_stop=stop,
                  buzz_model=buzz_model, buzz_tok=buzz_tok, device=device, tts=tts)


def _load_tts(device: str = "cuda", precision: str = "bf16"):
    """Irodori-TTS-500M-v3 の InferenceRuntime をロード（失敗時は None＝音声なしで継続）。"""
    try:
        from huggingface_hub import hf_hub_download
        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
        ckpt = hf_hub_download(repo_id="Aratako/Irodori-TTS-500M-v3", filename="model.safetensors")
        prec = precision if device == "cuda" else "fp32"
        rt = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=ckpt, model_device=device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
            model_precision=prec, codec_device=device, codec_precision=prec))
        print("[round_builder] Irodori TTS loaded")
        return rt
    except Exception as e:  # noqa: BLE001 — TTS は任意。失敗しても本体は動かす。
        print(f"[round_builder] ⚠ TTS load 失敗（音声なしで継続）: {type(e).__name__}: {e}")
        return None


# ユーザー確認の採用パラメータ（[[irodori-tts-integration]]）: 生テキスト（pykakasi不要）・num_steps=64。
def _synth_audio(tts, text: str, ref_wav: str, num_steps: int = 64, target_sr: int = 24000):
    """1問の読み上げ音声を合成し data:audio/wav;base64 文字列で返す（失敗時 None）。"""
    try:
        import base64
        import io
        from irodori_tts.inference_runtime import SamplingRequest
        res = tts.synthesize(SamplingRequest(
            text=text, ref_wav=ref_wav, no_ref=False,
            num_candidates=1, num_steps=int(num_steps), trim_tail=True))
        audio = res.audios[0].squeeze(0).float().cpu()
        sr = res.sample_rate
        if target_sr and target_sr != sr:
            import torchaudio
            audio = torchaudio.functional.resample(audio, sr, target_sr)
            sr = target_sr
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio.numpy(), sr, format="WAV", subtype="PCM_16")
        return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:  # noqa: BLE001
        print(f"[round_builder] ⚠ TTS synth 失敗（この問題は音声なし）: {type(e).__name__}: {e}")
        return None


# ============================================================
# 推論ヘルパー
# ============================================================
def _buzz_confs(m: Models, texts, batch_size, max_seq_length):
    import torch
    confs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = texts[i:i + batch_size]
            enc = m.buzz_tok(b, return_tensors="pt", padding=True, truncation=True,
                             max_length=max_seq_length).to(m.device)
            logits = m.buzz_model(**enc)
            confs.extend(torch.sigmoid(logits.float()).reshape(-1).tolist())
    return confs


def _find_buzz_pos(m: Models, question: str, theta: float, stride: int,
                   batch_size: int, max_seq_length: int) -> tuple[int, bool, list[dict]]:
    """char-stream を stride 走査し conf≥θ の初出位置を返す（無交差なら全長・False）。
    第3返り値は実測の確信度カーブ [{f: 位置/全長, c: conf}]（frontend のライブメーター用）。"""
    L = len(question)
    start = max(10, int(0.15 * L))
    positions = list(range(start, L + 1, max(1, stride)))
    if not positions or positions[-1] != L:
        positions.append(L)
    confs = _buzz_confs(
        m, [BUZZ_USER_TEMPLATE.format(n=p, prefix=question[:p]) for p in positions],
        batch_size, max_seq_length)
    curve = [{"f": round(p / L, 3), "c": round(c, 3)} for p, c in zip(positions, confs)]
    for p, c in zip(positions, confs):
        if c >= theta:
            return p, True, curve
    return L, False, curve


def _main_generate(m: Models, prefix: str, max_new_tokens: int, think: bool):
    """corpus-2 書式の user を chat に渡し `<think>…</think>answer` を生成。(answer, think文字列)。"""
    import torch
    user = MAIN_USER_TEMPLATE.format(n=len(prefix), prefix=prefix)
    msgs = [{"role": "user", "content": user}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        inputs = m.main_tok.apply_chat_template(msgs, enable_thinking=think, **kw)
    except TypeError:
        inputs = m.main_tok.apply_chat_template(msgs, **kw)
    inputs = inputs.to(m.main_model.device)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = m.main_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=m.main_eos,
                                    eos_token_id=m.main_stop or m.main_eos)
    gen = m.main_tok.decode(out[0][in_len:], skip_special_tokens=True)
    return _parse_gen(gen)


def _parse_gen(gen: str) -> tuple[str, str]:
    """生成テキストから (answer, think文字列) を取り出す。
    `</think>` が無い＝思考が閉じ切らなかった（ループ等）→ 回答は空（思考文を回答欄に出さない）。"""
    if "</think>" not in gen:
        return "", gen.replace("<think>", "").strip()
    think_txt = gen.split("</think>")[0].replace("<think>", "").strip()
    tail = gen.split("</think>")[-1]
    answer = tail.strip().splitlines()[0].strip() if tail.strip() else ""
    return answer, think_txt


def _main_generate_batch(m: Models, prefixes: list[str], max_new_tokens: int, think: bool):
    """複数 prefix を1回の generate でバッチ生成（左パディング）。
    [(answer, think), ...] を返す。ZeroGPU 窓を使い切るための主バッチ経路。"""
    import torch
    if not prefixes:
        return []
    tok = m.main_tok
    texts = []
    for p in prefixes:
        msgs = [{"role": "user", "content": MAIN_USER_TEMPLATE.format(n=len(p), prefix=p)}]
        try:
            s = tok.apply_chat_template(msgs, enable_thinking=think,
                                        add_generation_prompt=True, tokenize=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        texts.append(s)
    # デコーダ生成は左パディング必須（生成トークンが共通の in_len 以降に揃う）。
    old_side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(m.main_model.device)
    tok.padding_side = old_side
    in_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = m.main_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=m.main_eos,
                                    eos_token_id=m.main_stop or m.main_eos)
    return [_parse_gen(tok.decode(out[i][in_len:], skip_special_tokens=True))
            for i in range(len(prefixes))]


# ============================================================
# aiThink 整形（gemma <think> → [{frac, text}]・reading 中の段階表示用）
# ============================================================
_SENT_SPLIT = re.compile(r"(?<=[。！？\.!?])\s*")


def _think_steps(think_txt: str, buzz_frac: float, max_steps: int = 3,
                 mask: tuple = ()) -> list[dict]:
    """<think> を文分割し frac を ~0.2〜buzz_frac に均等配置（最大 max_steps 文）。

    mask: 解答候補の文字列群。読み上げ中に think を表示すると人間が答えを読めて
    しまう（カンニング）ため、各文の mask 文字列を ●● に置換した "masked" も併載する。
    frontend は判定前 masked / 判定後 text を表示する。"""
    if not think_txt:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(think_txt) if s.strip()]
    if not sents:
        return []
    # 長すぎる場合は末尾優先で max_steps 文に圧縮（決定的な手がかりは後半に出る）。
    if len(sents) > max_steps:
        sents = sents[-max_steps:]

    def _masked(s: str) -> str:
        for t in mask:
            if t and len(t) >= 2:
                s = s.replace(t, "●●")
        return s

    n = len(sents)
    lo, hi = 0.18, max(0.22, buzz_frac - 0.02)
    steps = []
    for i, s in enumerate(sents):
        frac = lo + (hi - lo) * (i / max(1, n - 1)) if n > 1 else hi
        steps.append({"frac": round(frac, 3), "text": s, "masked": _masked(s)})
    return steps


# ============================================================
# build_round — questions（{id, full, truth, category?, pattern?}）→ QUIZ_ROUND
# ============================================================
def build_round(questions: list[dict], m: Models, *, qutils,
                match: str = "Live Match", theta: float = 0.6,
                stride: int = 2, max_new_tokens: int = 256, think: bool = True,
                buzz_batch: int = 64, max_seq_length: int = 512,
                batch_main: bool = False, rebound: bool = True,
                tts_ref: str | None = None, tts_steps: int = 64,
                progress=None) -> dict:
    """各問について buzz 位置・回答・思考・正誤を実モデルで決め、QUIZ_ROUND dict を返す。

    questions: [{"id", "full", "truth", ("category"), ("pattern")}, ...]
    qutils: src/qutils（is_correct を使う・呼び出し側で import して渡す）。
    batch_main: gemma 生成を全問1バッチで回す（ZeroGPU 窓を使い切る・既定 True）。
    rebound: 人間が誤答した時に AI が全文で答え返す用の aiFullAnswer も precompute（既定 True）。
    progress: 任意 callable(i, n, qid) — Gradio 進捗表示等に使う。
    戻り値はフロントエンドの window.QUIZ_ROUND と同一スキーマ。
    """
    # ── pass 1: buzz 位置と prefix を全問ぶん用意 ──
    metas = []
    for i, q in enumerate(questions):
        full = q["full"]
        L = len(full)
        if progress:
            progress(i, len(questions), q.get("id"))
        buzz_pos, crossed, curve = _find_buzz_pos(m, full, theta, stride, buzz_batch, max_seq_length)
        metas.append({"q": q, "full": full, "L": L, "buzz_pos": buzz_pos,
                      "crossed": crossed, "prefix": full[:buzz_pos], "curve": curve,
                      "golds": q["truth"] if isinstance(q["truth"], list) else [q["truth"]]})

    # ── pass 2: gemma 生成（buzz地点回答 ＋ rebound 用の全文回答）をまとめてバッチ ──
    buzz_prefixes = [mm["prefix"] for mm in metas]
    full_prefixes = [mm["full"] for mm in metas] if rebound else []
    all_prefixes = buzz_prefixes + full_prefixes
    if batch_main:
        gens = _main_generate_batch(m, all_prefixes, max_new_tokens, think)
    else:
        gens = [_main_generate(m, p, max_new_tokens, think) for p in all_prefixes]
    buzz_gens = gens[:len(buzz_prefixes)]
    full_gens = gens[len(buzz_prefixes):] if rebound else [None] * len(metas)

    # ── 組み立て ──
    out_qs = []
    for i, (mm, (answer, think_txt)) in enumerate(zip(metas, buzz_gens)):
        L = mm["L"]
        buzz_frac = round(min(0.99, mm["buzz_pos"] / L), 4)
        correct = qutils.is_correct(answer, mm["golds"], loose=True)
        # think マスク対象＝AIの解答と正解（reading 中の表示で人間が答えを読めないように）。
        mask_strs = tuple({s for s in (answer, mm["golds"][0]) if s})
        rec = {
            "id": mm["q"].get("id", i + 1),
            "category": mm["q"].get("category", ""),
            "pattern": mm["q"].get("pattern", ""),
            "genre": mm["q"].get("genre", ""),
            "full": mm["full"],
            "truth": mm["golds"][0],
            "truthKana": mm["q"].get("truth_kana", ""),  # かな解答の判定救済（frontend judge 用）
            "buzzer": "ai",          # AI は buzz_frac で押す。human は live で先押し可（engine 側）。
            "buzzFrac": buzz_frac,
            "answer": answer,
            "correct": bool(correct),
            "aiThink": _think_steps(think_txt, buzz_frac, mask=mask_strs),
            "confCurve": mm["curve"],    # buzz 回帰ヘッドの実測確信度カーブ（ライブメーター用）
            "aiCrossed": mm["crossed"],  # θ 未交差（自信不足で全文まで行った）かの内部フラグ
        }
        if rebound and full_gens[i] is not None:
            fa, _ = full_gens[i]
            rec["aiFullAnswer"] = fa
            rec["aiFullCorrect"] = bool(qutils.is_correct(fa, mm["golds"], loose=True))
        out_qs.append(rec)

    # ── TTS: 各問の問題文を読み上げ（Irodori・生テキスト・参照音声クローン）──
    if m.tts is not None and tts_ref:
        for rec in out_qs:
            rec["audio"] = _synth_audio(m.tts, rec["full"], tts_ref, tts_steps)

    return {"match": match, "theta": theta, "questions": out_qs}
