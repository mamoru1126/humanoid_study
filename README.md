# humanoid_study

ヒューマノイドと、ロボットに作業を学習させる技術について、調べたことの資料と、試すための環境を置くリポジトリ。

公開ページ：https://mamoru1126.github.io/humanoid_study/

## 資料

| 日付 | 資料 |
| --- | --- |
| 2026-11-14 | [VLM・VLA の数理を学ぶロードマップ｜学習、推論、蒸留を数式で説明できるようになるまで](https://mamoru1126.github.io/humanoid_study/vla-math/) |
| 2026-11-14 | [Genie Sim を RTX 3090 で動かす｜仕組み、すごいところ、勉強になるところ](https://mamoru1126.github.io/humanoid_study/genie-sim/) |

## 環境

| 日付 | 環境 |
| --- | --- |
| 2026-11-14 | [Genie Sim を RTX 3090 で動かす環境（Genie Sim 3.6.0、Isaac Sim 6.2、Docker）](genie_sim_rtx3090/) |

環境は、手順書と bash スクリプトからなる。RTX 3090 の実機では、まだ動かしていない。確認できていることと、できていないことは、手順書の最初に書いてある。

## 演習コード

| 日付 | コード |
| --- | --- |
| 2026-11-14 | [VLM・VLA の数理：学習・推論・蒸留を 1 次元の問題で確かめる（numpy と matplotlib だけ。GPU は不要）](vla_math/) |

演習コードは、実行して確かめてある。確かめた環境は、`vla_math/README.md` に書いてある。

## メモ

GitHub Pages の公開元が、リポジトリの一番上（root）になっている。資料の本体は `docs/` にあるので、一番上の `index.html`、`genie-sim/index.html`、`vla-math/index.html` で `docs/` へ転送している。公開元を `/docs` に変えたあとは、この 3 つは使われないので消してよい。
