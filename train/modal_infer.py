"""Modal の安価GPU（T4/L4）で メインLLM SFT 済みモデルの推論を数件テストする。

目的: corpus-2 val から数件サンプリングし、訓練と同一の chatml 整形で
`<think>…</think>{answer}` を生成 → `</think>` 以降を answer として取り出し、
`qutils.is_correct` で正解判定する。生成ベース正解率（≥65%ゲート）の最小版。

設計の正典は docs/quiz-ai.md（Phase 1）/ train/README.md。推論整形は sft.py と一致させる。

GPU/メモリ（9B）:
  - L4 24GB + bf16  : merged 相当（base bf16 + LoRA）が ~18GB で載る。品質の基準（推奨）。
  - T4 16GB + 4bit  : QLoRA 訓練と同条件（base 4bit + LoRA）。~7GB。--load-4bit 必須。

実行:
  # L4・bf16・20件（既定）
  modal run train/modal_infer.py
  # T4・4bit（QLoRA 訓練と同条件）・10件
  modal run train/modal_infer.py --gpu T4 --load-4bit --n 10
"""

import os

import modal

app = modal.App("quiz-infer")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))  # qutils.is_correct を再利用

# 訓練（modal_sft.py）と同じ依存・キャッシュ。unsloth が Qwen3.5(qwen3_5) を patch。
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unsloth",
        "transformers",
        "datasets",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/hf",  # 訓練と同じ hf-cache Volume を再利用（base 再DLなし）
    })
    # src/qutils.py（正規化・正解判定）を bake。問題文を含まないので image に焼いてよい。
    .add_local_dir(_SRC, "/opt/quizsrc", copy=True,
                   ignore=["__pycache__", "*.pyc"])
)

SRC_DIR = "/opt/quizsrc"

corpus_vol = modal.Volume.from_name("quiz-corpus", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image,            # ⚠ 必須（未指定だとデフォルト image で unsloth/qutils 不在）
    gpu="L4",               # with_options(gpu=...) で T4 等に上書きする
    volumes={"/data": corpus_vol, "/hf": hf_cache},
    secrets=secrets,
    timeout=60 * 30,
)
def run(repo: str, n: int, load_4bit: bool, max_new_tokens: int, seed: int,
        variant: str, source: str, adapter_path: str = "", fewshot_k: int = 4):
    import json
    import random
    import sys

    sys.path.insert(0, SRC_DIR)
    import qutils  # is_correct / normalize（訓練・合成と同一の照合）

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from transformers import set_seed

    set_seed(seed)

    # アダプタを base+LoRA としてロード（unsloth が peft で base を自動取得）。
    # adapter_path 指定時は Volume 上のチェックポイント（例: 途中停止した epoch ベスト）を
    # 直接ロードする。未指定なら HF の {repo}-lora を使う。
    # load_4bit=False → bf16（L4・品質基準） / True → 4bit（T4・QLoRA訓練と同条件）。
    # source=base は SFT/アダプタなしの素のベース本体を読む（CPT土台の素知識比較）。
    # それ以外は adapter（{repo}-lora か Volume チェックポイント）を base+LoRA で読む。
    adapter_src = adapter_path or (repo if source == "base" else f"{repo}-lora")
    print(f"[infer] load {adapter_src}  4bit={load_4bit}  gpu="
          f"{os.environ.get('MODAL_GPU', '?')}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_src,
        max_seq_length=2048,
        load_in_4bit=load_4bit,
        dtype=None,
        token=os.environ.get("HF_TOKEN"),  # private repo
    )
    if source != "base":
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")  # sft.py と一致
    FastModel.for_inference(model)

    import torch

    def split_answer(content: str) -> str:
        # `<think>…</think>{answer}` の </think> 以降を answer とする。
        if "</think>" in content:
            return content.split("</think>", 1)[1].strip()
        return content.strip()

    def generate(user_content: str):
        """user メッセージ（早押しクイズ整形）を1件生成し (pred_head, think) を返す。"""
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        # Qwen3.5 は VLM のため tokenizer は processor。位置引数だと images に
        # 渡り画像ロードを試みて落ちる → text= を明示してテキスト経路に通す。
        inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy（再現性）
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        pred = split_answer(gen)
        pred_head = pred.splitlines()[0].strip() if pred else ""  # answer は基本1語
        think = (gen.split("</think>", 1)[0].replace("<think>", "").strip()
                 if "</think>" in gen else "(no </think>)")
        return pred_head, think

    # corpus-2 と同じ user 整形（sft.py の訓練分布に合わせる）
    def fmt_user(char_pos: int, text: str) -> str:
        return f"早押しクイズ（{char_pos}文字目時点）:\n{text}"

    if source == "base":
        # SFT/アダプタなしの素のベースモデルの「生」知識を few-shot で測る。
        # 目的: CPT土台候補（Qwen3.5-9B vs gemma-4-12B 等）の素のJPトリビア知識を比較。
        # few-shot 例は train split、評価対象は val split（held-out）→ リーク無し。
        path = "/data/annotated_questions.jsonl"
        train_pool, val_pool = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("is_valid"):
                    continue
                (val_pool if qutils.qid_split(r["qid"]) == "val"
                 else train_pool).append(r)
        rng = random.Random(seed)
        rng.shuffle(train_pool)
        rng.shuffle(val_pool)
        shots = train_pool[:fewshot_k]
        targets = val_pool[:n]
        shot_text = "".join(
            f"問題: {s['question']}\n答え: {s['answers'][0]}\n\n" for s in shots)
        header = "以下はクイズの問題と答えです。答えは簡潔に1つだけ書きます。\n\n"
        print(f"[infer] base few-shot: model={adapter_src} k={fewshot_k} "
              f"評価{len(targets)}件")

        def raw_answer(question: str) -> str:
            prompt = header + shot_text + f"問題: {question}\n答え:"
            inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            return gen.strip().splitlines()[0].strip() if gen.strip() else ""

        ok = ok_l = 0
        for i, r in enumerate(targets, 1):
            golds = r["answers"]
            pred = raw_answer(r["question"])
            c = qutils.is_correct(pred, golds)
            cl = qutils.is_correct(pred, golds, loose=True)
            ok += c
            ok_l += cl
            mark = "✅" if c else ("➕" if cl else "❌")
            print(f"[{i}/{len(targets)}] {mark} gold={golds} pred={pred!r}")
        N = len(targets) or 1
        print(f"\n[infer] === base素知識(few-shot) N={N}  model={adapter_src} ===")
        print(f"  全文正解率: strict {ok}/{N}={ok / N:.1%}  /  "
              f"loose {ok_l}/{N}={ok_l / N:.1%}")
        return

    if source == "paired":
        # 同一問題に base(素・few-shot) と SFT(think→answer) の両方を答えさせ突き合わせる。
        # 「base✓ sft✗ = SFTが知識を取りこぼし(elicitation損失)」「両✗ = 真の知識欠落」を切り分け。
        # 1モデル(base+LoRA)をロードし、base答えは disable_adapter() で素のbaseに戻して生成。
        assert hasattr(model, "disable_adapter"), (
            "Peftモデルでなく adapter を無効化できない＝ペアにならない。"
            "--adapter-path か {repo}-lora を指定せよ。")
        path = "/data/annotated_questions.jsonl"
        train_pool, val_pool = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("is_valid"):
                    continue
                (val_pool if qutils.qid_split(r["qid"]) == "val"
                 else train_pool).append(r)
        # val選択は seed のみで決定（評価集合を固定）。few-shot例は別RNGで選び評価集合を乱さない。
        random.Random(seed).shuffle(val_pool)
        targets = val_pool[:n]
        shots = train_pool[:]
        random.Random(seed + 1).shuffle(shots)
        shots = shots[:fewshot_k]
        shot_text = "".join(
            f"問題: {s['question']}\n答え: {s['answers'][0]}\n\n" for s in shots)
        header = "以下はクイズの問題と答えです。答えは簡潔に1つだけ書きます。\n\n"
        print(f"[infer] paired: model={adapter_src} few-shot k={fewshot_k} "
              f"評価{len(targets)}件（同一問題で base vs SFT）")

        def base_answer(question: str) -> str:
            prompt = header + shot_text + f"問題: {question}\n答え:"
            inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad(), model.disable_adapter():
                out = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            return gen.strip().splitlines()[0].strip() if gen.strip() else ""

        bL = sL = bS = sS = 0          # base/sft の loose(L)・strict(S) 正解数
        both = only_base = only_sft = neither = 0  # 2x2（loose基準）
        for i, r in enumerate(targets, 1):
            golds = r["answers"]
            q, L = r["question"], r["question_length"]
            pred_b = base_answer(q)
            pred_s, _ = generate(fmt_user(L, q))   # SFT think→answer（adapter有効）
            bok = qutils.is_correct(pred_b, golds, loose=True)
            sok = qutils.is_correct(pred_s, golds, loose=True)
            bL += bok
            sL += sok
            bS += qutils.is_correct(pred_b, golds)
            sS += qutils.is_correct(pred_s, golds)
            if bok and sok:
                both += 1
            elif bok:
                only_base += 1
            elif sok:
                only_sft += 1
            else:
                neither += 1
            tag = ("両✓" if bok and sok else "base✓sft✗" if bok
                   else "sft✓base✗" if sok else "両✗")
            print(f"[{i}/{len(targets)}] {tag}  gold={golds}  "
                  f"base={pred_b!r}  sft={pred_s!r}")
        N = len(targets) or 1
        print(f"\n[infer] === ペア base vs SFT  N={N} ===")
        print(f"  base素(few-shot) 正解: loose {bL}/{N}={bL / N:.1%}  / "
              f"strict {bS}/{N}={bS / N:.1%}")
        print(f"  SFT(think→answer) 正解: loose {sL}/{N}={sL / N:.1%}  / "
              f"strict {sS}/{N}={sS / N:.1%}")
        print(f"  内訳(loose): 両✓={both}  base✓sft✗(elicitation損失)={only_base}  "
              f"sft✓base✗={only_sft}  両✗(知識欠落)={neither}")
        print("  → base✓sft✗が多い=SFTが取りこぼし(引き出し改善で回収可) / "
              "両✗が多い=真の知識欠落(CPT/大Base)")
        return

    if source == "full":
        # 全文診断（ペア）: annotated_questions を val split で読み、同一問題を
        # 「全文」と「buzz位置(relaxed=buzz_char+5)」の両方で評価し正解率を比較する。
        # 全 answers リストで照合（corpus-2 の単一 gold より正確）。
        path = "/data/annotated_questions.jsonl"
        pool = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("is_valid") and qutils.qid_split(r["qid"]) == "val":
                    pool.append(r)
        random.Random(seed).shuffle(pool)
        rows = pool[:n]
        print(f"[infer] full診断: annotated val split {len(pool)}件 → ペア評価{len(rows)}件")

        # strict（従来）と loose（早押し読み上げ相当の緩和）の両方を集計。
        full_ok = buzz_ok = recovered = 0
        full_ok_l = buzz_ok_l = recovered_l = 0
        for i, r in enumerate(rows, 1):
            q, L, bc = r["question"], r["question_length"], r["buzz_char"]
            golds = r["answers"]
            fpred, fthink = generate(fmt_user(L, q))                       # 全文
            bprefix = q[:min(bc + 5, L)]
            bpred, _ = generate(fmt_user(bc, bprefix))                     # buzz位置(relaxed)
            fok = qutils.is_correct(fpred, golds)
            bok = qutils.is_correct(bpred, golds)
            fok_l = qutils.is_correct(fpred, golds, loose=True)
            bok_l = qutils.is_correct(bpred, golds, loose=True)
            full_ok += fok
            buzz_ok += bok
            full_ok_l += fok_l
            buzz_ok_l += bok_l
            if fok and not bok:
                recovered += 1
            if fok_l and not bok_l:
                recovered_l += 1
            # loose のみ正解は (loose) と付記
            fm = "✅" if fok else ("➕" if fok_l else "❌")
            bm = "✅" if bok else ("➕" if bok_l else "❌")
            print(f"\n[{i}/{len(rows)}] qid={r['qid']}  全文{fm} / buzz{bm}  "
                  f"buzz_ratio={r.get('buzz_ratio')}  (➕=looseのみ正解)")
            print(f"  gold      : {golds}")
            print(f"  full_pred : {fpred}")
            print(f"  buzz_pred : {bpred}  (prefix末尾: …{bprefix[-18:]})")
            print(f"  full_think: {fthink[:110]}")
        N = len(rows)
        print(f"\n[infer] === 全文ペア診断 N={N}  (strict / loose) ===")
        print(f"  全文正解率   : {full_ok}/{N}={full_ok / N:.1%}  /  "
              f"{full_ok_l}/{N}={full_ok_l / N:.1%}")
        print(f"  buzz正解率   : {buzz_ok}/{N}={buzz_ok / N:.1%}  /  "
              f"{buzz_ok_l}/{N}={buzz_ok_l / N:.1%}")
        print(f"  buzz失敗→全文で救済: strict {recovered}/{N - buzz_ok}  /  "
              f"loose {recovered_l}/{N - buzz_ok_l}")
        print(f"  → 全文≫buzz なら位置要因(設計通りGRPO再校正へ) / "
              f"全文も低いなら知識要因(SFT強化)")
        return

    # source == "val": corpus-2 val を読む。variant フィルタ後に seed 固定で
    # シャッフルして n 件（先頭固定だと qid 順で出題ソースが偏るため代表性を確保）。
    #   variant=exact   : question[:buzz_char]（早い切り出し＝難）
    #   variant=relaxed : question[:buzz_char+5]（5文字後・やや易）
    #   variant=all     : 両方
    val_path = "/data/sft_corpus_2/val.jsonl"
    all_rows = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if variant != "all" and ex.get("meta", {}).get("variant") != variant:
                continue
            all_rows.append(ex)
    random.Random(seed).shuffle(all_rows)
    rows = all_rows[:n]
    print(f"[infer] val {val_path}: variant={variant} 該当{len(all_rows)}件 → 評価{len(rows)}件")

    correct = correct_loose = 0
    for i, ex in enumerate(rows, 1):
        user_msg = ex["messages"][0]                 # role=user の部分問題文
        gold = split_answer(ex["messages"][1]["content"])
        pred_head, think = generate(user_msg["content"])
        ok = qutils.is_correct(pred_head, [gold])
        ok_l = qutils.is_correct(pred_head, [gold], loose=True)
        correct += ok
        correct_loose += ok_l

        qid = ex.get("meta", {}).get("qid", "?")
        prefix = user_msg["content"].split("\n", 1)[-1]
        mark = "✅" if ok else ("➕" if ok_l else "❌")  # ➕=looseのみ正解
        print(f"\n[{i}/{len(rows)}] qid={qid}  {mark}")
        print(f"  prefix : {prefix[:60]}")
        print(f"  gold   : {gold}")
        print(f"  pred   : {pred_head}")
        print(f"  think  : {think[:120]}")

    N = len(rows) or 1
    print(f"\n[infer] 正解 strict {correct}/{len(rows)}={correct / N:.1%}  /  "
          f"loose {correct_loose}/{len(rows)}={correct_loose / N:.1%}  "
          f"(参考ゲート ≥65% / ➕=looseのみ)")


@app.local_entrypoint()
def main(repo: str = "YUGOROU/quiz-main-sft", gpu: str = "L4", n: int = 20,
         load_4bit: bool = False, max_new_tokens: int = 320, seed: int = 3407,
         variant: str = "all", source: str = "val", adapter_path: str = "",
         fewshot_k: int = 4):
    # GPU は with_options で上書き（T4 16GB なら --load-4bit 必須）。
    # source=val   : corpus-2 val prefix を評価（variant=relaxed で早buzz除外）
    # source=full  : 全文ペア診断（annotated_questions val split を全文+buzzで比較）
    # source=base  : SFTなしの素ベースを few-shot 全文QAで評価（CPT土台の素知識比較）。
    #   repo にベースモデルID を渡す（-lora を付けない）。bf16の12Bは --gpu A100-40GB。
    #   modal run train/modal_infer.py --source base --n 100 \
    #     --repo Qwen/Qwen3.5-9B-Base --gpu A100-40GB
    #   modal run train/modal_infer.py --source base --n 100 \
    #     --repo google/gemma-4-12B --gpu A100-40GB
    # adapter-path : Volume上のチェックポイント直接評価（例: 途中停止した epoch ベスト）
    #   modal run train/modal_infer.py --source full --n 50 \
    #     --adapter-path /data/out/main_sft_v2/checkpoint-384
    fn = run.with_options(gpu=gpu)
    fn.remote(repo=repo, n=n, load_4bit=load_4bit, max_new_tokens=max_new_tokens,
              seed=seed, variant=variant, source=source, adapter_path=adapter_path,
              fewshot_k=fewshot_k)
