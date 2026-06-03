# Phase 0b 実測ハーネス

RTX 5090 32GB（Vast.ai）上で、早押しクイズAIのレイテンシ・速度を実測する。
特に**割り込みモデル LFM2.5-350M が「毎文字 Reasoning」を char 到着間隔(~100ms想定)に
収められるか**を検証するのが主目的。設計の正典は `docs/quiz-ai.md` / `docs/corpus.md`、
直近の意思決定は `docs/HANDOFF.md`。

## 3つの実測

| # | スクリプト | 何を測る | 状態 |
|---|---|---|---|
| ② | `bench_lfm2_decode.py` (HF) / `bench_lfm2_decode_vllm.py` (vLLM) | LFM2.5-350M の decode速度・毎文字thinkレイテンシ。budget内に載る think トークン数 | ✅ 作成済 |
| ③ | `bench_qwen_colocate.py` | Qwen3.5-9B(vLLM)投機推論と同居時の buzz レイテンシ劣化（優先ストリーム/MPS有無） | ⬜ 未作成 |
| ① | `bench_parakeet_char.py` | parakeet 実機の新かな確定間隔（=②③のbudgetの根拠） | ⬜ 未作成（要サンプル音声） |

実行順は **② → ③**（②の素の数字が③の基準値）。①は音声サンプル準備後。

## 確定している前提（HANDOFF / メモリ参照）
- **メインLLM = Qwen/Qwen3.5-9B-Base**：vLLM 0.17+ がネイティブ対応（`qwen3_5`, GDN+MoE hybrid）。
  - **vision除外** = vLLM の `language_model_only=True`（自前実装不要）。
  - **罠: FP8 KVキャッシュ禁止**（gibberish劣化）。重みはFP8でよいが `--kv-cache-dtype` は bf16/fp16。
- **割り込み = LiquidAI/LFM2.5-350M-Base**（text-only, 16層=10conv+6GQA, KVはGQA6層のみ＝増分に有利）。
- VRAM 18–21GB / 32GB で余裕。レイテンシ予算は新char間隔 ~100ms。

## セットアップ（5090インスタンス上）
```bash
# uv 導入済み前提。HF_TOKEN を export（モデルDL用）
export HF_TOKEN=...        # 値はログに出さない
# モデルは初回 generate/LLM() 時に自動DL。disk は 50GB+ 推奨（Qwen 9B 級が乗る）
```

## 実測②の回し方
```bash
# HF transformers版（素のネイティブ速度・カスタムランタイムの目安）
uv run bench/bench_lfm2_decode.py --compile --attn flash_attention_2 \
    --budget-ms 100 --out /tmp/lfm2_hf.json

# vLLM版（APC増分prefill込み・本番候補）
uv run bench/bench_lfm2_decode_vllm.py --budget-ms 100 --out /tmp/lfm2_vllm.json
```
**見るべき出力**:
- `[pure-decode] N tok/s` … per-token decode 時間。`think N tok ≈ (N+1)/tps` の基準。
- `[sequential-buzz]` の各 think_tokens 行の **p90** が `budget` 以下なら ○。
- `[判定]` … budget(100ms)内に載る think トークン数（実測p90 と 増分換算理論）。
- think_tokens=0 は「回帰ヘッド相当（生成なし1 forward）」の下限レイテンシ目安。

**解釈**: HF版の逐次模擬は prefill フル再計算の上界寄り。本番の KV 増分では増分換算値に近づく。
vLLM版は APC で共通プレフィックスを再利用するので、両者の差が「増分の効き」を示す。

## 実測③の設計（未作成・実装時の指針）
- Qwen3.5-9B-Base を vLLM で常駐（`language_model_only`, weight FP8, **KV bf16**, enable_prefix_caching）。
- 投機推論を模擬（部分プレフィックスから生成を走らせ続ける）しつつ、
  同一GPUで LFM2.5-350M の buzz 判定レイテンシを測る。
- **比較軸**: (a) LFM2.5単独, (b) Qwen同居・優先ストリームなし, (c) 同居・高優先度CUDAストリーム, (d) NVIDIA MPS。
- 知りたいのは「Qwen の重い生成中に buzz 判定が SM 待ちでどれだけ劣化するか」（VRAMでなく SM 時間共有が本丸）。

## ベンチ対象モデル（HF, 2026-06 実在確認済み）
- `LiquidAI/LFM2.5-350M-Base` / `LiquidAI/LFM2.5-350M`（instruct）
- `Qwen/Qwen3.5-9B-Base`（VLM, vision除外して使用）
- `nvidia/parakeet-tdt_ctc-0.6b-ja`, `Aratako/Irodori-TTS-500M-v3`, Silero VAD
