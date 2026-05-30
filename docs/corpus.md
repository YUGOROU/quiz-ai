# 早押しLMシステム — コーパス前処理 実装指示書

## 概要

早押しクイズ特化LLMシステムの訓練コーパスを3種類生成するパイプライン。
全パイプラインはPython、実行環境はmacOS (M3) またはGoogle Colab。

```
AI王 V2.0 (22,335問)
    │
    ├─[1. 共通前処理]─ annotated_questions.jsonl (~17,000問)
    │                         │
    │              ┌──────────┼──────────┐
    │              ↓          ↓          ↓
    │         corpus-1    corpus-2   corpus-3
    │         (~27万件)   (~3.4万件)  (~2.2万件)
    └─[未フィルタ分]──────────────────────→ corpus-3に追加
```

---

## データソース

全てAmazon S3から直接ダウンロード（認証不要）。

| ファイル | URL |
|---|---|
| V2.0 学習用 (22,335問) | `https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_train.jsonl` |
| V2.0 開発用 (1,000問) | `https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_dev_v1.0.jsonl` |
| V3.0 dev (500問・prefix形式) | `https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_04/aio_04_dev_v1.0.jsonl` |

**V2.0フォーマット:**
```json
{"qid": "QA20QBIK-0002", "question": "童謡『たなばたさま』の歌詞で...", "answers": ["ササ"]}
```

**V3.0フォーマット（1文字ずつprefix）:**
```json
{"qid": "AIO04-0001", "position": 1, "question": "1"}
{"qid": "AIO04-0001", "position": 44, "question": "1945年、ラテンアメリカの...誰?"}
```

V3.0には正解ラベルが別途必要（`aio_04_dev_v1.0.jsonl` の answers フィールド参照）。

---

## Step 1: 共通前処理（バズアノテーション）

### 目的
各問題の「統計的確定点（S-buzz）」を特定し `annotated_questions.jsonl` を生成する。
corpus-1・2はこのファイルを入力とする。

### フィルタ条件（アノテーション前）

| 条件 | 値 |
|---|---|
| 問題文長 | ≥ 50文字 |
| 正解の文字数 | ≥ 2文字 |
| 正解が問題文に含まれていない | 完全一致で除外 |

### バズアノテーションアルゴリズム

問題文の長さ L に対して、binary search で以下を求める：
「LLM が k 回中 r 回以上正答できる最小文字位置 buzz_char」

```
パラメータ:
  model   = deepseek-v4-flash             (Crof.ai経由。フォールバック: openai/gpt-oss-120b:cerebras)
  k       = 5                             (1ポジションあたりの試行数。時間/コストが厳しければk=3)
  threshold = 0.8                         (5回中4回以上正答)
  min_pos = max(10, L * 0.15)             (最小探索開始位置)

LLMへの入力プロンプト:
  "以下は早押しクイズの問題文の途中（{n}文字目まで）です。\n問題: {prefix}\n答えは1語で。不明な場合は「不明」と答えてください。\nAnswer:"

正規化関数（日本語表記ゆれ対応）:
  unicodedata.normalize("NFKC", text)
  括弧内の読み仮名を除去: re.sub(r'[（(][^）)]*[）)]', '', text)
  末尾の敬称を除去: (さん|くん|氏|博士|先生)
```

### フィルタ条件（アノテーション後）

| 条件 | 値 |
|---|---|
| `is_valid` | True（全文でもLLMが答えられない問題を除外） |
| `buzz_ratio` | `[0.20, 0.85]` |

### 出力スキーマ（annotated_questions.jsonl）

```json
{
  "qid": "QA20QBIK-0002",
  "question": "問題文全文",
  "answers": ["ササ"],
  "question_length": 52,
  "buzz_char": 31,
  "buzz_ratio": 0.596,
  "confidence_curve": {"10": 0.05, "26": 0.41, "31": 0.82, "42": 0.94, "52": 0.98},
  "is_valid": true
}
```

### 注意事項
- API呼び出しは並列化する（`concurrent.futures.ThreadPoolExecutor`）。**deepseek-v4-flash は単ストリーム~89 t/s だが、合成はオフラインバッチなので並列度で隠せる**。`max_workers` を 10→30→50 と上げて頭打ち点（Crof.ai同時接続上限）を探る
- **本番前に200問でパイロット**を回し、(a) 実集約スループット、(b) buzz_char の妥当性、(c) 壁時計時間の外挿を確認。許容外なら高volumeのcorpus-1のみ gpt-oss にハイブリッド
- Step1は全工程で**1モデルに統一**すること（混ぜると buzz_char の基準がぶれて分布が壊れる）
- 中断再開のため、処理済み qid をキャッシュファイルに逐次書き出す

### キャリブレーション・ギャップ（重要な注意）
S-buzz は「**アノテモデルが答えられる位置**」。deepseek-v4級（高性能）でアノテすると、buzz_char が**デプロイ先メインLLM(9B)が実際に答えられるより早い位置**に付きうる。デフォルトは「buzz_char を理想（オラクル）ターゲットと割り切り、**GRPOで実機性能に再キャリブレーション**」とし、評価フェーズでアノテ基準と実機正解位置の乖離を必ずモニタする。

---

## Step 2: SFT-corpus-1（バズタイミングモデル用）

### 入力
- `annotated_questions.jsonl`（Step 1出力）
- AI王 V3.0 dev（prefix形式、別途処理）

### サンプリング戦略

1問から最大16件のprefixをサンプリングする。

```
区分A（buzz周辺・密）: buzz_char ± 10文字 を2文字刻み → 最大10件
区分B（前半・疎）:     [10文字, buzz_char×0.33, buzz_char×0.66] → 3件
区分C（後半・疎）:     [buzz_char×1.2, buzz_char×1.5, L文字] → 3件
重複除去・範囲外クリップ後に確定
```

### 信頼度ラベル計算

```python
import math

def confidence(n, buzz_char, question_length):
    steepness = 15.0 / question_length
    return 1.0 / (1.0 + math.exp(-steepness * (n - buzz_char)))
```

`confidence_curve` に実測値がある位置は補正:
`final = 0.6 × sigmoid_value + 0.4 × measured_value`

### `<think>` 生成

`deepseek-v4-flash`（Crof.ai）に以下を投げて1〜2文の短い内部独白を生成させる:
```
入力: 問題文プレフィックス（{n}文字目まで）+ 正解
指示: 「このプレフィックスから答えを推定する思考過程を1〜2文で。解答候補と根拠のみ。」
```
割り込みモデル（LFM2.5-350M）は速度クリティカルなので think は常に短く保つ（1〜2文上限）。

### 出力スキーマ（1件）

```json
{
  "messages": [
    {"role": "user", "content": "問題文（{n}文字目まで）:\n{prefix}"},
    {"role": "assistant", "content": "<think>{short_reasoning}</think>{confidence:.2f}"}
  ],
  "meta": {
    "qid": "QA20QBIK-0002",
    "position": 31,
    "buzz_char": 31,
    "confidence_label": 0.82,
    "is_buzz_region": true
  }
}
```

### 推定サイズ
17,000問 × 16件 ≈ **272,000件**

---

## Step 3: SFT-corpus-2（クイズ回答LLM用）

### 入力
- `annotated_questions.jsonl`（Step 1出力）

### prefix入力の2バリアント

各問題につき2件生成する:
- `variant=exact`: `question[:buzz_char]`
- `variant=relaxed`: `question[:buzz_char + 5]`（末尾切り詰め不要）

### reasoning chain生成（adaptive thinking）

`deepseek-v4-flash`（Crof.ai）に推論を生成させる。**reasoning長を難易度（buzz_ratio）で変える**:
```
入力: プレフィックス + 正解 + 目標think_mode
指示（full）  : 「プレフィックスの手がかりのみを使い、正解に至る推論を3〜5文で。最後に正解を出力。」
指示（short） : 「手がかりから1〜2文で簡潔に推論し、正解を出力。」
指示（none）  : 「推論なしで正解のみ出力。」（<think></think> は空）
```

think_mode の割り当て方針:
- buzz_ratio が小さい（早buzz＝簡単）→ short / none を多めに
- buzz_ratio が大きい（遅buzz＝難問）→ full
- 思考量変調を学習させるため、同一問題の一部を別 think_mode の変種としても生成する

推論時は max think tokens で上限を切り、強制 `</think>` で answer フェーズへ。

### 出力スキーマ（1件）

```json
{
  "messages": [
    {"role": "user", "content": "早押しクイズ（{buzz_char}文字目時点）:\n{prefix}"},
    {"role": "assistant", "content": "<think>{reasoning}</think>{answer}"}
  ],
  "meta": {
    "qid": "QA20QBIK-0002",
    "buzz_char": 31,
    "variant": "exact",
    "think_mode": "full",
    "think_budget": 256
  }
}
```

### 推定サイズ
17,000問 × 2件 ≈ **34,000件**（think_mode変種を加える場合はその分増）

---

## Step 4: RL-corpus（GRPO用）

### 入力
- AI王 V2.0 train 全量（22,335問）
- アノテーション成功分は `buzz_char_reference` を付与

### 出力スキーマ（1件）

```json
{
  "prompt": "問題文全文",
  "answer": "ササ",
  "meta": {
    "qid": "QA20QBIK-0002",
    "question_length": 52,
    "buzz_char_reference": 31,
    "buzz_ratio_reference": 0.596
  }
}
```

アノテーション失敗問題は `buzz_char_reference: null` で含める。

### 報酬関数仕様（実装参照用）

正式な報酬定義は `quiz-ai.md` の Phase 2「報酬関数（確定）」を参照（対戦相手の機会損失・スルー込み、両モデル共同責任）。1問単位の基本形:

```
R(buzz, answer, correct, length):
  if normalize(answer) != normalize(correct): return -1.5   # 誤答
  position_score = 1.0 - (buzz / length)
  return 1.0 + 0.5 × position_score        # 正解（早いほど高、1.0〜1.5）

# 対戦エピソードでは追加で:
#   相手に先取りされた  : -0.5
#   答えられたのにスルー : -0.1
#   ガードレール: |機会損失| < |誤答| を維持
```

### 推定サイズ
**22,335件**

---

## Train / Val / Test 分割

**必ず qid 単位で分割すること**（prefix単位で分割すると同一問題が train/test に混在しリークが発生）。

```
split比率: train 0.85 / val 0.10 / test 0.05
方法: qid のハッシュ値で決定論的に分割（シード固定不要）
適用: corpus-1・2・3 で同一の qid 分割を使い回す
```

---

## ファイル構成（推奨）

```
corpus/
├── raw/
│   ├── aio_02_train.jsonl
│   ├── aio_02_dev.jsonl
│   └── aio_04_dev.jsonl
├── annotated_questions.jsonl      # Step 1 出力・キャッシュ兼用
├── annotation_cache.jsonl         # 処理済みqidと結果（中断再開用）
├── sft_corpus_1/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── sft_corpus_2/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── rl_corpus/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

---

## 実行方法（uv）

各スクリプトの先頭に以下のインラインメタデータを記述し、`uv run` で実行する。

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "tqdm",
# ]
# ///
```

```bash
uv run annotate.py
uv run build_corpus1.py
# ...
```

venv・requirements.txt・pip install は不要。標準ライブラリ + `openai` + `tqdm` のみで完結させること（pandas・numpy 不使用推奨）。

---

## API設定（Crof.ai / deepseek-v4-flash）

```python
from openai import OpenAI

# 本命: Crof.ai deepseek-v4-flash
client = OpenAI(
    base_url=os.environ["CROFAI_BASE_URL"],   # Crof.ai の OpenAI互換エンドポイント
    api_key=os.environ["CROFAI_API_KEY"],
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[...],
    max_tokens=64,
)

# フォールバック: HF Inference Providers gpt-oss-120b:cerebras
# base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"],
# model="openai/gpt-oss-120b:cerebras"
```

環境変数: `CROFAI_API_KEY`, `CROFAI_BASE_URL`（フォールバック用に `HF_TOKEN`）

---

## コスト

単価: deepseek-v4-flash $0.12/$0.21、gpt-oss-120b:cerebras $0.25/$0.69（フォールバック）。
総コール数は約100万回（Step1のbinary search × k=5 が支配的）。

### 実測（2026-05-29 パイロット、annotate.py --limit 200）
- **Step1アノテ ≈ $0.0032/問**（100問で$0.318）。机上見積り($17)の約6倍。
- 主因の疑い：**deepseek-v4-flash が reasoning（思考）トークンを出力**し課金されている（Step1は答え1語しか要らないのに）。
- 未最適化のままの外挿：Step1 ~$60 / Step2 ~$30-40 / Step3 ~$5-7 → **合計 ~$100**。

### コスト最適化（Phase 1再開時に適用）
1. **思考トークン抑制（最重要）**：合成モデルのデフォルト reasoning を最小化（`reasoning_effort=minimal` 等のAPIパラメータ／プロンプト指示）。
   - Step1（アノテ）：reasoning は**完全に不要** → 答えのみ出力させる。
   - Step2/3：欲しい reasoning は**可視の出力**として書かせる（隠れ思考は捨てるだけで訓練価値ゼロ）。「問題文＋buzz位置を与え、割り込み/回答モデルのReasoningを出力内で演じさせる」方針。
2. **`response.usage` ロギング**で prompt/completion/reasoning トークン内訳を確認してから本番投入。
3. コール数削減：Step1 `k=5→3`（約40%減）、Step2 prefix 16→10、Step3 簡単問題を `none`モード化。
4. 見込み：上記で **~$100 → ~$30-40** 以下。

> 注: GitHub Copilot/GitHub Models は無料枠の日次上限（Low 150/day, High 50/day）＋本番利用禁止のため、本バッチには使用不可（`quiz-ai.md` 参照）。

---

## 実装上の注意

- Rate limit: 429 エラー時は exponential backoff で再試行。`max_workers` はパイロットで決めた頭打ち点に設定
- 中断再開: `annotation_cache.jsonl` を逐次 append、起動時に既処理 qid を set に読み込む
- 正規化は全工程で統一した関数を使う（`utils/normalize.py` に切り出し推奨）

## ライセンス（重要）

- 生成物（`annotated_questions.jsonl`, `sft_corpus_*/`, `rl_corpus/`）は **AI王の問題文を含むため公開・再配布しない**（private厳守）。
- 公開してよいのは学習済みモデル重み＋推論コードのみ。詳細と帰属表記は `quiz-ai.md` の「ライセンス」節を参照。
