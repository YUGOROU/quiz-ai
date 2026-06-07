# Phase 0b — E2E serving / 接続

ライブ早押しパイプラインのバックエンド配線。**学習・評価は完了済**で、ここは
「gemma メイン＋buzz 回帰ヘッドを serve し、orchestrator から叩く」結線。

設計の正典は `docs/quiz-ai.md` / `docs/HANDOFF.md`。本 README は手順のみ。

## 構成

```
[MacBook]                         [Vast 5090]
  問題テキスト ──事前TTS同期──▶ char schedule ─┐
  (会場再生)                                   ▼
                              p0_orchestrator.run_episode
                                ├─ buzz_decider ──HTTP──▶ serve_buzz.py  :8001 (回帰ヘッド)
                                └─ MainLLM.answer ─HTTP──▶ vllm serve     :8000 (gemma)
  AI回答テキスト ◀──WS over Tailscale── 回答確定
  (MLX Irodori でローカル合成)
```

WAN ハンデ消去の核心（HANDOFF「E2E接続アーキ」）= **事前計算TTS＋同期再生＋
タイムスタンプ判定**。問題音声 uplink と parakeet STT は問題経路では不要化、
buzz は T_ai/T_human のタイムスタンプ＋整定窓で会場ハブ(MacBook)が権威判定。

## Vast 側セットアップ

```bash
# 軽量公式イメージ pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime + --ssh(proxy)
export HF_TOKEN=hf_...                 # read 権限可
bash serve/setup_vast.sh              # apt build-essential + pip vllm + モデルDL
# (Tailscale も張る場合) WITH_TAILSCALE=1 bash serve/setup_vast.sh

bash serve/serve_main.sh &            # gemma → :8000 (OpenAI互換, stop 1,106)
uv run serve/serve_buzz.py &          # buzz 回帰ヘッド → :8001

curl localhost:8000/v1/models         # メイン疎通
curl localhost:8001/health            # buzz 疎通
uv run src/buzz_client.py "743年に聖武天皇が"   # conf 確認
```

## orchestrator から実行

```bash
# メインを Vast vLLM serve に向ける（localhost 同居 or Tailscale IP）
export P0_MAIN_BASE_URL=http://127.0.0.1:8000/v1
export P0_MAIN_MODEL=quiz-main
# buzz を回帰ヘッド serve に差し替え（src/buzz_client.make_buzz_decider）
# → p0_run_eval / 新 E2E ランナーで run_episode(..., buzz_decider=...) に注入
uv run src/p0_run_eval.py --in corpus/annotated_questions.jsonl --limit 50
```

`run_episode(buzz_decider=make_buzz_decider(theta=θ))` で buzz を注入する。
θ は速度↔精度ノブ: S-buzz ちょうど(θ≈0.45)で main ~62%、θ を上げて打点を
+4〜8字 遅らせると 69〜74%（メイン側ゲート ≥65% クリア）。

## 計測すべきメトリクス（Phase 0b 第一級）

- buzz確定 → 音声出力開始 **≤ 1.0s** / 新char到着 → buzz判定 **≤ 300ms**
- メイン投機推論 TTFT・total、buzz前に投機完了した割合、キャンセル健全性
- （同期再生採用後）クロック同期精度・char-schedule整合・buzz整定窓の妥当性

## 注意

- buzz は別プロセス serve（GIL回避・CLAUDE.md 設計）。同期 decider は localhost
  HTTP（~数ms）。WAN/高負荷で律速するなら async 化（orchestrator seam を async に拡張）。
- FP8 を使う場合のみ vLLM 公式 Docker（deep_gemm 同梱）。既定は bf16/4bit で回避。
- インスタンス作成は課金発生 = 都度ユーザー承認。残高・disk(50GB+)・CN除外に注意。
- UI（`~/Downloads/design_handoff_quiz_buzzer_ai/`）統合は別途ユーザー指示まで未着手。
