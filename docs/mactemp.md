# mactemp

MacのCPU/GPU温度と消費電力をsudoなしで取得するzsh関数。Claude Codeがリモートビルド中の発熱を確認するために使用する。

## 依存

- [macmon](https://github.com/vladkens/macmon) (`brew install macmon`)
- [jq](https://jqlang.org) (`brew install jq`)

## 使用方法

```bash
mactemp
```

### 出力例

```json
{
  "cpu": 36,
  "gpu": 34,
  "cpu_w": 0.09
}
```

| フィールド | 説明 | 単位 |
|---|---|---|
| `cpu` | CPU平均温度 | ℃ |
| `gpu` | GPU平均温度 | ℃ |
| `cpu_w` | CPU消費電力 | W |

## 温度の目安

| 状態 | CPU温度 |
|---|---|
| アイドル | 35〜45℃ |
| ビルド中（正常） | 60〜75℃ |
| 要注意 | 85℃以上（サーマルスロットリング開始） |

## Claude Codeでの使用指針

- 長時間ビルド前後に `mactemp` で温度確認する
- CPU温度が85℃を超えた場合はビルドを一時停止し、ユーザーに報告する
- 継続的な監視が必要な場合は `watch -n 5 mactemp` を使用する（Ctrl+Cで停止）
