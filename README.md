# quiz-ai — 早押しクイズ特化型LLMシステム

競技早押しクイズに特化した2モデル構成のAIシステム。問題文を音声でリアルタイム受信し、
人間と同条件（ハンデなし）で早押し対戦できるレベルを目指す。

差別化軸: **Reasoning獲得 / ゲーム理論的戦略 / クイズ文法学習 / GRPO / 音声入力**

## 構成

```
quiz-ai/
├── docs/                  設計ドキュメント
│   ├── quiz-ai.md         アーキテクチャ・設計判断・フェーズ・ライセンス
│   ├── corpus.md          コーパス前処理 実装指示書
│   └── mactemp.md         Mac発熱監視ユーティリティ
└── src/                   実装
    ├── qutils.py          共通: 正規化・LLMクライアント・qid分割・I/O
    ├── annotate.py        Step1: S-buzzアノテーション
    ├── build_corpus1.py   Step2: 割り込みモデル用コーパス
    ├── build_corpus2.py   Step3: メインLLM用コーパス
    ├── build_rl_corpus.py Step4: GRPO用コーパス
    ├── p0_llm.py          Phase0: メインLLM 非同期ストリーミング
    ├── p0_orchestrator.py Phase0: asyncioオーケストレータ
    └── p0_run_eval.py     Phase0a: テキストF/S評価
```

## 2モデル構成

| 役割 | モデル | 機能 |
|---|---|---|
| 割り込みモデル | LiquidAI/LFM2.5-350M | 毎文字 buzz/no-buzz 判定（Reasoning採用） |
| メインLLM | Qwen3.5-9B-Base（本命）/ gemma-4-E4B | 部分問題文から投機的に回答予測 |
| STT | parakeet-tdt_ctc-0.6b-ja | CTC Streaming（文字単位出力） |
| TTS | Irodori-TTS-500M-v3 | 音声合成 |
| VAD | Silero VAD | 無音検出 |

## 現在のフォーカス: Phase 0（パイプラインF/S）

- **0a（GPU不要）**: テキストのみで投機推論の打ち切り機構・メインLLMのprefix正解率・レイテンシを検証。
- **0b（Colab T4）**: parakeet STT + Silero VAD + Irodori TTS を同ランタイムに統合しE2E計測。

実行手順は [`src/README.md`](src/README.md) を参照。

## データ・ライセンス（重要）

- 訓練データは AI王 (Project AIO) / JAQKET 由来。**問題文を含む生成物（`corpus/`）は公開・再配布しない**（`.gitignore` 済み）。
- 公開可能なのは学習済みモデル重み＋推論コードのみ（帰属表記つき）。詳細は [`docs/quiz-ai.md`](docs/quiz-ai.md) のライセンス節。
- 合成API: Crof.ai `deepseek-v4-flash`（フォールバック HF `gpt-oss-120b:cerebras`）。
