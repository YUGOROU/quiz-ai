# 早押しクイズLM — 実装スクリプト

`docs/corpus.md`（コーパス前処理）と `docs/quiz-ai.md`（Phase 0）の実装。
AI王 V2.0 (22,335問) から3種のコーパスを生成し、パイプラインF/Sを検証する。

| ファイル | Step | 役割 | LLM呼び出し |
|---|---|---|---|
| `qutils.py` | 共通 | 正規化・クライアント・qid分割・I/O | — |
| `annotate.py` | 1 | S-buzz アノテーション → `annotated_questions.jsonl` | 多（binary search × k=5）|
| `build_corpus1.py` | 2 | 割り込みモデル用（~27万件）| 多（think生成）|
| `build_corpus2.py` | 3 | メインLLM用（~3.4万件、adaptive thinking）| 中 |
| `build_rl_corpus.py` | 4 | GRPO用（22,335件）| なし |

合成モデル: **Crof.ai deepseek-v4-flash**（フォールバック: HF `gpt-oss-120b:cerebras`）。

## Google Colab での実行

```python
# 1. 取得（private repo を PAT でクローン）
#    Colab Secrets に GH_PAT（repoスコープのPAT）を登録しておく
from google.colab import userdata
PAT = userdata.get("GH_PAT")
!git clone https://{PAT}@github.com/YUGOROU/quiz-ai.git
%cd quiz-ai/src

# 2. 依存（tqdmは標準で入っている。openaiのみ）
!pip -q install openai

# 3. APIキー（Colab Secrets に CROFAI_API_KEY を登録 → 環境変数へ）
import os
from google.colab import userdata
os.environ["CROFAI_API_KEY"] = userdata.get("CROFAI_API_KEY")
# base_url は既定で https://crof.ai/v1、model は deepseek-v4-flash（変更時のみ設定）
# os.environ["CROFAI_BASE_URL"] = "https://crof.ai/v1"
# os.environ["SYNTH_MODEL"]     = "deepseek-v4-flash"
# os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # フォールバックを使う場合

# 4. まずパイロット（200問で並列度・妥当性・所要時間を確認）
!python annotate.py --limit 200 --max-workers 32

# 5. 本番（中断したら同じコマンドで cache から再開）
!python annotate.py --max-workers 40
!python build_corpus1.py --max-workers 40
!python build_corpus2.py --max-workers 40
!python build_rl_corpus.py
```

## ローカル / uv

```bash
export CROFAI_API_KEY=...        # base_url は既定 https://crof.ai/v1
uv run annotate.py --limit 200
```

## 重要な注意

- **並列度**: deepseek-v4-flash は単ストリーム ~89 t/s だが、バッチ合成なので `--max-workers` を 10→30→50 と上げて頭打ち点（Crof.ai 同時接続上限）を探る。
- **中断再開**: 各スクリプトは `*_cache.jsonl` に逐次追記。Colab Free は切断されやすいので、再実行で自動再開する設計。
- **ライセンス**: 生成物（`annotated_questions.jsonl`, `sft_corpus_*`, `rl_corpus`）は **AI王の問題文を含むため公開・再配布しない**。公開可なのは学習済みモデル重み＋推論コードのみ（帰属表記つき）。詳細は `docs/quiz-ai.md` のライセンス節。
- **キャリブレーション注意**: S-buzz はアノテモデルが答えられる位置。デプロイ先メインLLM(9B)より早い可能性があり、buzz_char は理想ターゲットとして扱い GRPO で再キャリブレーションする。

## Phase 0: パイプラインF/S検証

コーパス合成とは独立。既存モデルでパイプラインの妥当性とレイテンシを検証する。

| ファイル | 役割 |
|---|---|
| `p0_llm.py` | メインLLM（gpt-oss-120b:cerebras / HF）非同期ストリーミング＋キャンセル対応 |
| `p0_orchestrator.py` | asyncioオーケストレータ（char供給・ルールbuzz・投機推論・コミット）。char-source/buzzは差し替え可 |
| `p0_run_eval.py` | 100問でF/S計測（正解率・buzz→回答レイテンシ・投機ヒット率） |

### Phase 0a（テキストのみ・GPU不要）

```python
# Colab T4 / Mac。メインLLMはHF Inference経由なのでGPU不要。
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")     # gpt-oss-120b:cerebras

# まず少数で正解率確認（fastで実時間シミュレートを省略）
!python p0_run_eval.py --limit 20 --fast
# 投機の先行効果も含めた本計測（実時間シミュレート）
!python p0_run_eval.py --limit 100
# キャンセル機構の自己テスト（前提ズレを全問注入）
!python p0_run_eval.py --limit 10 --reset-frac 1.0
```

出力 `phase0a_report.json` に正解率・`buzz_to_answer_s`・`spec_done_before_buzz_rate` 等。
成功基準: 正解率 >= 0.60、buzz→回答 <= 1.0s。

主な調整: `--buzz-ratio 0.65` `--spec-lead 8`（buzz何文字手前で投機開始）`--char-rate 12`。

### Phase 0b（音声・Colab T4で同ランタイム）

`p0_orchestrator.py` の char-source を `sim_char_source` から parakeet STT 出力に、
buzz を rule から LFM2.5-350M に差し替える（同インターフェース）。T4 16GB に
parakeet(~1.2GB)+VAD+Irodori-TTS(~1GB) は同居可能。メインLLMはAPIのままVRAM不要。
（0a の数値確定後に実装）

## 出力構成

```
corpus/
├── raw/aio_02_train.jsonl          # 自動DL
├── annotation_cache.jsonl          # Step1 全件（再開用）
├── annotated_questions.jsonl       # Step1 フィルタ後
├── corpus1_cache.jsonl / corpus2_cache.jsonl
├── sft_corpus_1/{train,val,test}.jsonl
├── sft_corpus_2/{train,val,test}.jsonl
└── rl_corpus/{train,val,test}.jsonl
```
