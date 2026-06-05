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
  - 実装は `train/modal_infer.py`（`--source val|full|base|paired`）。

## CPT（継続事前学習）— メインLLM の知識天井対策

**動機（docs/HANDOFF.md）**: メインLLM の全文正解率は **48% が 9B の知識ハード上限**。
ハイパラ/SFTデータ増量/elicitation のいずれでも動かないとペア評価で実証決着 →
parametric 知識を上げる **CPT が唯一のレバー**。日本語 wiki で CPT し、**CPT→再SFT→全文eval**
で 48% が動くかを検証する。RAG・27B は却下、gemma-4-12B は Unsloth 未対応で土台は Qwen3.5-9B-Base に据置。

**コーパス**: `range3/wiki40b-ja`（wiki40b＝Google 品質フィルタ版の日本語抽出・CC-BY-SA・「幅広いが小さい」）。
全量 ~0.9–1.5B tok 規模なので **token-budget でサブサンプル**（既定 100M tok）して run 時間/コストを制御。

**手法 = full-param FT-CPT（確定 2026-06-05）**: LoRA の低ランク容量限界は知識注入に不利なため、
Modalクレジット$250 獲得（無料枠/コスト制約が消えた）を受けて**全パラメータ FT** を選択（`--full-ft`）。
H100 80GB で `Trainable 9.41B/9.41B (100%)`・bf16 full FT（メモリ50%減）・peak 60.5/79GiB(bs=2)。
GPU=**H100**（本ワークロードは248K語彙lm_headで帯域律速＝帯域最大GPUが最速 / tok/$ 微最適化は不要に）。

**SFT との違い**（`train/cpt.py`）: 素の CLM（chatml なし・response-only マスクなし）／full FT は
`embed_tokens`/`lm_head` も自動で学習対象／`embedding_learning_rate` を本体 LR の 1/10（5e-5 / 5e-6）。
QLoRA 経路（`--full-ft` 無し）も残置（r=128＋embed/lm_head、低コスト比較用）。

⚠ **packing は事前パッキングで対処**（`build_cpt_corpus.py` の `--pack-seq 2048`）。Qwen3.5 は VLM(processor系)で
Unsloth の sample packing が無効化される（`Sample packing skipped`）→ コーパス側で token粒度に 2048tok ブロックへ
連結し padding 浪費を消す。各行が再tokenizeで 2040–2048tok の密系列になる（検証済）。

```bash
# 1) コーパス整形（Modal CPU ジョブが Volume /cpt_corpus へ直接生成・packing込み。ローカルDL/put不要）
#    初回 full-FT 信号run は 50M tok。step時間はサイズ非依存なので疎通は同コーパスの先頭10stepで足る。
uv run --with modal modal run train/modal_cpt.py::prep --token-budget 50_000_000 --pack-seq 2048
#    ローカルで整形したい場合: uv run src/build_cpt_corpus.py --out-dir ~/quiz-ai-corpus/corpus
#      → uv run --with modal modal volume put quiz-corpus .../cpt_corpus /cpt_corpus

# 2) full-FT-CPT（まず疎通で step 時間/メモリ較正 → 12h timeout に収まるか確認）
#    entrypoint が prep/main の2つあるので ::main を明示
uv run --with modal modal run train/modal_cpt.py::main --full-ft --max-steps 10 --batch-size 2
uv run --with modal modal run train/modal_cpt.py::main --full-ft --push-to-hub YUGOROU/quiz-qwen-cpt
#   → YUGOROU/quiz-qwen-cpt-merged（bf16 全重み）が再SFT の土台になる

# 3) 再SFT（CPT 済みモデルを base に corpus-2 を再 SFT）
uv run --with modal modal run train/modal_sft.py --target main --run-tag cpt \
  --base-model YUGOROU/quiz-qwen-cpt-merged --push-to-hub YUGOROU/quiz-main-sft-cpt

# 4) 全文eval で 48% が動いたか確認
uv run --with modal modal run train/modal_infer.py --source full \
  --repo YUGOROU/quiz-main-sft-cpt --n 200 --gpu A100-40GB
```

⚠ full-FT は固定コスト（weights+grads+adam8bit）≈54GB＋活性。bs はメモリ実測で決める（bs=2 で 76%）。
12h timeout を超えそうなら token-budget を落とすか checkpoint-resume（`--max-steps` 無しで save_strategy=epoch）。
