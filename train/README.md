# SFT 訓練（Phase 1）

早押しクイズAIの2モデルを Unsloth + TRL で SFT する。設計の正典は
`docs/quiz-ai.md`（Phase 1 節）/ `docs/corpus.md`。

| モデル | base | コーパス | 出力形式 | max_seq |
|---|---|---|---|---|
| 割り込み | LiquidAI/LFM2.5-350M-Base | corpus-1 (`sft_corpus_1`) | `<think>…</think>{confidence:.2f}` | 512 |
| メインLLM | Qwen/Qwen3.5-9B-Base | corpus-2 (`sft_corpus_2`) | `<think>…</think>{answer}` | 2048 |

- Base モデルに **chatml テンプレートを後付け**し、`<|im_start|>assistant\n` 以降のみで損失を取る
  （`train_on_responses_only`）。**推論時も同じ chatml 整形を使うこと**。
- `max_seq_length` はコーパス実測の char 長 p99/max から決定（buzz: p99=174/max=330、main: p99=340/max=866）。
- LoRA rank/alpha は CLAUDE.md で未決 → 既定 r=alpha=32。`sft.py` の `CONFIGS` で調整。

## 移行基準（SFT→GRPO・quiz-ai.md 確定）

- 割り込み: buzz位置 **MAE ≤ 8文字**（val）
- メイン: corpus-2 val の **answer正解率 ≥ 65%**
- 両方クリアで Phase 2 (GRPO) へ。

## GPU と依存

訓練は GPU 機（**Modal A100 40GB** = 無料枠 / **Vast.ai RTX 5090** = 本番収束）。**Mac では動かない**。
依存: `unsloth`, `trl>=0.12`, `datasets`, `transformers`, `torch`（CUDA）。

### A) Modal A100（推奨・無料枠）

```bash
uv tool install modal && modal setup        # ブラウザ認証
modal volume create quiz-corpus
modal volume put quiz-corpus ~/quiz-ai-corpus/corpus/sft_corpus_2 /sft_corpus_2
# corpus-1 完走後: modal volume put quiz-corpus ~/quiz-ai-corpus/corpus/sft_corpus_1 /sft_corpus_1

# 疎通run（10step・課金最小。設定の妥当性を安く検証）
modal run train/modal_sft.py --target main --max-steps 10
# 本番run
modal run train/modal_sft.py --target main
modal volume get quiz-corpus /out/main_sft ./outputs/main_sft
```

無料枠は月~14h。`main`(9B QLoRA) を優先し、`buzz`(350M) は軽いので Vast/ローカルでも可。
**支出上限**: ワークスペースの spend limit に達すると `App creation failed` で即死する
（Modal ダッシュボード Settings → Usage で引き上げ）。

#### 疎通run 実測（2026-06-04・A100 40GB）

- ✅ unsloth 2026.6.1 が `Qwen3_5` をネイティブ patch、**OOMなし**で 9B QLoRA がA100 40GBに載る。
- ✅ LoRA trainable 58.2M / 9.47B (0.61%)、`TARGET_MODULES` 標準集合で通過。
- ✅ chatml + response-only 損失で loss 1.90→1.32 と低下。
- 速度 **~9.6s/step（3.3 samples/s）** → corpus-2 本番 ~767step ≈ **約2時間**。
- モデル重みは `hf-cache` Volume に永続化済み → 次回 run は**再DLなし**。
- ⚠ `image=image` を `@app.function` に必ず渡すこと（未指定だとデフォルト image になり
  unsloth/sft 不在で `ModuleNotFoundError`）。

### B) GPU 機で直接（Vast.ai 5090 等）

```bash
pip install unsloth "trl>=0.12" datasets
python train/sft.py --target buzz --data-root ~/quiz-ai-corpus/corpus --out-root outputs
python train/sft.py --target main --data-root ~/quiz-ai-corpus/corpus --out-root outputs
# HF Private へ push する場合: --push-to-hub <org>/quiz-main-sft
```

## 既知の注意・要実機確認

- **target_modules**: 標準の attn+MLP 集合。LFM2 の conv 層 / Qwen3.5 の Gated DeltaNet・sparse MoE
  expert で射影名が異なりエラーになる場合、`sft.py` の `TARGET_MODULES` を `"all-linear"` に置換。
- **Qwen3.5-9B-Base**: vLLM 0.17+ / 新しめの Unsloth が必要。**KVキャッシュ FP8 は品質劣化で禁止**
  （メモリ `main-llm-qwen35-9b-vllm`）。SFT 時は QLoRA(4bit) 重みで A100 40GB に収める。
- **corpus-1 は生成完走待ち**（think 形式は Phase 0b 後に最終確定の可能性／HANDOFF 参照）。
  完走前に buzz を回すと部分データになる。
- **ライセンス**: コーパス由来の重みは **HF Private** のみ。公開は推論コードと、帰属表記つきモデルカード
  （`docs/quiz-ai.md` ライセンス節）。`--push-to-hub` は内部で `private=True`。

## SFT 後の評価（別途・未実装）

- 割り込み: val prefix で confidence を出し、閾値交差位置と buzz_char の **MAE** を測る。
- メイン: corpus-2 val(`variant=exact`) で `<think>` 打ち切り後の answer 正解率（`src/qutils.is_correct` 流用）。
