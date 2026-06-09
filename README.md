# quiz-ai — 早押しクイズ特化型LLMシステム

競技早押しクイズ（hayaoshi quiz）に特化した2モデル構成のAIシステム。問題文を1文字ずつ受信し、
人間と同条件（ハンデなし）で「いつ押すか」を判断して早押し対戦する。日本語特化。

差別化軸: **Reasoning獲得 / ゲーム理論的戦略 / クイズ文法学習 / 早押しタイミングの学習**

> ライブデモ（HF Space・ZeroGPU）は `space/` に同梱。本リポジトリは**学習・評価・デモのコードのみ**を公開し、
> 問題文を含むコーパス・デモ問題プールは再配布しない（下記ライセンス節）。

## 2モデル構成（合計 ≤ 32B）

| 役割 | モデル | 機能 |
|---|---|---|
| 割り込み（buzz） | `YUGOROU/quiz-buzz-reg-1.2bjp-merged`（LFM2.5-1.2B + 回帰ヘッド） | 問題文を毎文字読み confidence を出力、`conf ≥ θ` で buzz。約9ms/char |
| メインLLM | `YUGOROU/quiz-main-gemma-merged`（gemma-4-26B-A4B SFT） | buzz時点の部分問題文から `<think>…</think>` で推論し回答 |
| STT | parakeet-tdt_ctc-0.6b-ja | CTC Streaming（文字単位出力・設計本命/デモではオプション） |
| TTS | Irodori-TTS-500M-v3 | 問題文の読み上げ |
| VAD | Silero VAD | 無音検出 |

両モデルとも AI王 / JAQKET 由来のクイズ文法コーパスで SFT 済（コーパスは非公開、公開は重み＋推論コードのみ）。

## リポジトリ構成

```
quiz-ai/
├── docs/         設計ドキュメント（quiz-ai.md / corpus.md / mactemp.md）
├── src/          コーパス前処理・Phase0 オーケストレータ・共通ユーティリティ
│   ├── qutils.py            正規化・採点(is_correct)・LLMクライアント・qid分割
│   ├── annotate.py          Step1: S-buzz アノテーション
│   ├── build_corpus{1,2}.py Step2/3: 割り込み用 / メイン用コーパス生成
│   ├── build_rl_corpus.py   Step4: GRPO 用コーパス
│   ├── label_genres.py      デモ問題プールのジャンル分類（LLM）
│   ├── p0_orchestrator.py   Phase0: asyncio 投機推論オーケストレータ
│   ├── p0_llm.py / buzz_client.py  メインLLM / buzz 判定クライアント
│   └── p0_run_eval.py       Phase0a テキストF/S評価
├── train/        訓練・評価（Modal 上で実行）
│   ├── sft.py / modal_sft.py       メイン・buzz の SFT
│   ├── buzz_reg.py / buzz_rl.py    buzz 回帰ヘッド / 単独RL
│   ├── eval_knowledge.py           知識天井・全文/prefix 正解率評価
│   ├── eval_buzz.py                buzz 位置 MAE 評価
│   ├── e2e_modal.py                E2E（buzz→メイン投機推論→採点）検証
│   └── check_gemma4_*.py           gemma-4 の SFT/vLLM 疎通確認
├── serve/        推論サーバ配線（buzz FastAPI / メイン vLLM serve）
├── bench/        速度ベンチ（LFM2.5 decode・問題文非含）
└── space/        HF Space（ZeroGPU + Gradio）ライブデモ
```

## モデル（Hugging Face）

| 用途 | リポジトリ |
|---|---|
| メイン（gemma-4-26B-A4B SFT・統合） | `YUGOROU/quiz-main-gemma-merged` |
| 割り込み（LFM2.5-1.2B 回帰ヘッド・統合） | `YUGOROU/quiz-buzz-reg-1.2bjp-merged` |

## 実行

- 訓練・評価は **Modal**（`uv run --with modal modal run train/…`）。
- ライブデモは **HF Space（ZeroGPU）** で `space/` をデプロイ（詳細は `space/README.md`）。
- スクリプトは `uv run` 前提（`python3` 直叩き不可）。詳細手順は各 `README.md` 参照。

## データ・ライセンス（重要）

- 訓練データは AI王 (Project AIO) / JAQKET 由来。**問題文を含む生成物（`corpus/`・`annotated_questions.jsonl`・
  デモ問題プール `questions_*.json` 等）は公開・再配布しない**（`.gitignore` 済み。`space/build_aio_pool.py` +
  `src/label_genres.py` でローカル再生成する）。
- 公開可能なのは学習済みモデル重み＋推論/学習コードのみ（帰属表記つき）。
  > Quiz questions © abc/EQIDEN実行委員会 / 株式会社キュービック / クイズ法人カプリティオ. Non-commercial research use only. No dataset redistribution.
- `space/irodori_tts/` は [Aratako/Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) を vendoring したもので、上流ライセンスに従う。
- 合成API: Crof.ai `deepseek-v4-flash`（フォールバック HF `gpt-oss-120b:cerebras`）。
