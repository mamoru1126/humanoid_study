# Genie Sim を RTX 3090 で動かす環境

AgiBot のシミュレーション基盤 [Genie Sim](https://github.com/AgibotTech/genie_sim) を、RTX 3090（VRAM 24 GB）を 1 枚積んだ PC で動かすための、手順書とスクリプトです。

Genie Sim の本体、アセット、Docker イメージは、このリポジトリには含めません。スクリプトが、決めた版を取得します。仕組みや、何が勉強になるかは、資料にまとめています。

資料：<https://mamoru1126.github.io/humanoid_study/genie-sim/>

この手順で目指すのは、次の 3 つを順に通すことです。

1. 方策なしで、ロボットのカメラ画像を 1 組保存する。
2. 「動かない方策」をつないで、評価を最後まで回し、録画する。
3. 2 台目の PC に置いた方策サーバを、ネットワーク越しにつなぐ。

## 確認できていること、できていないこと

**この環境は、RTX 3090 の実機ではまだ一度も動かしていません。** スクリプトは、GPU のない環境で、Genie Sim の本物のコードと、偽の `docker` と `nvidia-smi` を使って確かめたものです。

| 項目 | 状態 |
|---|---|
| Genie Sim の本体と、ロボットのモデルの取得 | 確認済み。スクリプトで実際に取得した |
| アセットの取得範囲（`g2op_if` で 10 タスク、6.42 GiB） | 確認済み。Genie Sim の本物の CLI で計算した。ダウンロードそのものは未確認 |
| コンテナを起動するコマンドの組み立て | 確認済み。本物の CLI と起動スクリプトを偽の `docker` で動かし、引数を確かめた |
| スクリプトの流れと、失敗したときの止まり方 | 確認済み。偽の `docker` と `nvidia-smi` で、成功と失敗の両方を通した |
| ビルド済みイメージの取得と、その大きさ | 未確認 |
| RTX 3090 での Isaac Sim 6.2 の起動と描画 | 未確認。NVIDIA の動作確認の一覧に RTX 3090 はない |
| カメラ画像の保存、評価、録画 | 未確認 |
| VRAM の使用量と、かかる時間 | 未計測 |

最初の実行が、そのまま検証になります。うまくいかなかったときは、[記録を残す](#記録を残す) の手順で記録をまとめると、原因を調べやすくなります。

## 前提の版

| もの | 版 |
|---|---|
| Genie Sim | 3.6.0（コミット `083793f`、2026-11-13） |
| Isaac Sim | 6.2（コンテナの中） |
| コンテナの中 | Ubuntu 24.04、ROS 2 Jazzy、Python 3.12 |
| PC 側の OS | Ubuntu 22.04 か 24.04 |

PC 側に ROS 2 は要りません。必要なものはコンテナの中にそろっています。PC 側が Ubuntu 22.04 と ROS 2 Humble でも、そのまま使えます。

Genie Sim は更新が速いので、手順を確かめた版に固定しています。版を変えるときは、`.env` の `GENIESIM_REF` を書き換えます。

## RTX 3090 で気をつけること

- **NVIDIA の動作確認の一覧にありません。** Isaac Sim 6.2 の最小の GPU は RTX 4080（VRAM 16 GB）です。RTX 3090 は VRAM が 24 GB あり、RT コアも持っていますが、世代が 1 つ前です。動く見込みはありますが、保証はありません。
- **ドライバの版。** Isaac Sim 6.2 の動作確認は、Linux のドライバ 595.58.03 で行われています。手順 1 のスクリプトが、今の版と比べて表示します。
- **libuipc が使えません。** 布などの変形する物体のソルバ libuipc は、RTX 40 系と 50 系だけに対応しています。RTX 3090 では、これを使う 3 つのタスク（`g2crsbwuji_dexmanip_cloth_hanger`、`g2crsbwuji_dexmanip_drawstring_bag`、`g2crsbwuji_dexmanip_fold_cloth`）が動きません。ほかのタスクには影響しません。コンテナの起動のときに「libuipc unavailable」と出ますが、想定どおりです。
- **アセットは必要な分だけ取得します。** 全部で 17.70 GiB あります。既定では、指示どおりに物を取る 10 タスク（`g2op_if`、6.42 GiB）だけを取得します。
- **方策の推論は、別の PC に分けられます。** シミュレーションと方策は、ネットワークでつながる別のプログラムです。RTX 3090 が 2 台あるなら、1 台でシミュレーション、もう 1 台で方策を動かせます。

## 必要なもの

- GPU：RTX 3090（VRAM 24 GB）
- メモリ：32 GB 以上（Isaac Sim 6.2 の最小要件）
- ディスクの空き：Docker の置き場所に 60 GB 以上、作業用の置き場所に 30 GB 以上。イメージの大きさを確認できていないので、余裕を見た値です
- OS：Ubuntu 22.04 か 24.04
- NVIDIA のドライバ
- Docker と NVIDIA Container Toolkit（入っていなければ、手順 2 と 3 で入れます）
- git と、Python 3.10 から 3.12（Ubuntu 22.04 と 24.04 に最初から入っている python3 で足ります）
- GitHub、Docker Hub、Hugging Face につながるネットワーク

## 手順

コマンドは、1 つずつ実行します。どのスクリプトも、途中で失敗するとそこで止まり、「NG:」で始まる行に理由を表示します。やり直すときは、同じスクリプトをもう一度実行します。

| 手順 | スクリプト | やること | 取得する量 |
|---|---|---|---|
| 1 | `01_check_host.sh` | PC の状態を確かめる | なし |
| 2 | `02_install_docker.sh` | Docker を入れる（入っていなければ） | 小さい |
| 3 | `03_install_nvidia_toolkit.sh` | NVIDIA Container Toolkit を入れる（入っていなければ） | 小さい |
| 4 | `04_fetch.sh` | Genie Sim の本体を取得する | 約 1 GB |
| 5 | `05_assets.sh` | アセットを取得する | 6.42 GiB |
| 6 | `06_image.sh` | Docker イメージを用意する | 未確認。数十 GB の見込み |
| 7 | `07_up.sh` | コンテナを起動する | なし |
| 8 | `08_preview.sh` | カメラ画像を 1 組保存する | なし |
| 9 | `09_run_hold.sh` | 動かない方策で、評価を最後まで回す | なし |

### 0. このリポジトリを取得する

```bash
git clone https://github.com/mamoru1126/humanoid_study
```

```bash
cd humanoid_study/genie_sim_rtx3090
```

これ以降のコマンドは、すべてこのディレクトリで実行します。

設定は `.env` に書きます。最初にスクリプトを実行したときに、`.env.example` から自動で作られます。既定では、取得したものを `~/genie_sim_ws` に置きます。置き場所を変えたいときは、先に `.env` を作って、`GS_WORKDIR` を書き換えます。

```bash
cp .env.example .env
```

### 1. PC の状態を確かめる

```bash
bash scripts/01_check_host.sh
```

何も変更しません。OS、GPU、ドライバ、メモリ、ディスクの空き、Docker、NVIDIA Container Toolkit、git、Python、画面を調べて、項目ごとに「OK」「注意」「NG」を表示します。

- 最後に「NG: 0 件」と出れば、次へ進めます。
- 「NG」がある項目は、同じ行に直し方が出ます。Docker と NVIDIA Container Toolkit の NG は、手順 2 と 3 で直します。
- 「注意」は、読んで判断する項目です。RTX 3090 では、libuipc についての注意が必ず 1 件出ます。想定どおりです。
- Docker と NVIDIA Container Toolkit がすでに入っているときは、手順 4 へ進むように表示されます。

### 2. Docker を入れる（入っていなければ）

```bash
bash scripts/02_install_docker.sh
```

Docker の公式の apt リポジトリから入れます。`sudo` を使います。すでに `docker` が使えるときは、何もせずに終わります。

途中で、今の利用者を `docker` グループに加えるかを尋ねます。加えたときは、一度ログアウトして、ログインし直します。そのあとで、手順 1 をもう一度実行して確かめます。

### 3. NVIDIA Container Toolkit を入れる（入っていなければ）

```bash
bash scripts/03_install_nvidia_toolkit.sh
```

コンテナの中から GPU を使うための部品です。`sudo` を使います。最後に、試しのコンテナの中で `nvidia-smi` を実行して、GPU が見えることを確かめます。

### 4. Genie Sim の本体を取得する

```bash
bash scripts/04_fetch.sh
```

次の 3 つを行います。

- Genie Sim の本体を、決めた版だけ取得する（約 1 GB）。
- ロボットのモデル（サブモジュール）を取得する。
- `geniesim` コマンドを、Python の仮想環境（`~/genie_sim_ws/venv`）に入れる。PC 側の Python の環境は汚しません。

最後に「geniesim コマンドを入れました」と出れば成功です。

### 5. アセットを取得する

```bash
bash scripts/05_assets.sh
```

シーン、物体、ロボット、背景の画像を取得します。既定の範囲は `g2op_if`（10 タスク、6.42 GiB）です。取得する範囲の表示、ダウンロード、ファイルの検査の順に進みます。

- 途中で止まったときは、同じコマンドをもう一度実行します。続きから取得します。
- Hugging Face から取得できないときは、`.env` の `ASSET_SOURCE` を `modelscope` に変えます。
- 範囲を広げたいときは、`.env` の `ASSET_SELECTORS` を書き換えます。書き方は `.env.example` にあります。

アセットには、コードとは別のライセンスがあります。[ライセンス](#ライセンス) を読んでください。

### 6. Docker イメージを用意する

```bash
bash scripts/06_image.sh
```

既定では、Genie Sim の文書が案内しているビルド済みのイメージ（`yonhdee/genie_sim:geniesim4`）を取得して、Genie Sim が使う名前を付けます。大きさは確認できていませんが、Isaac Sim を含むので、数十 GB を見込んでください。

手元でビルドすることもできます。そのときは、`.env` の `IMAGE_MODE` を `build` に変えてから、同じスクリプトを実行します。RTX 3090 では、ビルドのときに libuipc の対象を指定する必要があります。`.env` の `LIBUIPC_SM=89` がその指定で、スクリプトが自動で渡します。

### 7. コンテナを起動する

```bash
bash scripts/07_up.sh
```

コンテナ `geniesim4` を起動して、準備が終わるまで待ちます。初回は、コンテナの中で Python のパッケージを登録するので、時間がかかります。待つ上限は 900 秒です（`.env` の `UP_TIMEOUT`）。

- 「コンテナの準備ができました」と出れば成功です。続けて、コンテナから見える GPU を表示します。
- 「libuipc unavailable」についての説明が出ますが、RTX 3090 では想定どおりです。
- 画面のある端末から実行すると、画面をつないだ状態で起動します。SSH 越しなど、画面がないときは、画面なしで起動します。どちらでも、この先の手順は進められます。

コンテナは、止めるまで動き続けます。止めるときは次を実行します。

```bash
bash scripts/down.sh
```

### 8. カメラ画像を 1 組保存する

```bash
bash scripts/08_preview.sh
```

方策をつながずに Isaac Sim を起動し、ロボットの頭、左手首、右手首のカメラ画像を 1 枚ずつ保存して終わります。Isaac Sim の起動、シーンの読み込み、描画までを確かめる手順です。

- 初回は、描画の準備（シェーダのコンパイル）に時間がかかります。2 回目からは、キャッシュが効きます。
- 「カメラ画像を 3 枚保存しました」と出れば成功です。保存した場所も表示します。
- 画像を開いて、机と物体とロボットの腕が写っていることを確かめます。
- 実行の間の VRAM の使用量の最大も表示します。

ここが、RTX 3090 で動くかどうかの最初の関門です。失敗したときは、[うまくいかないとき](#うまくいかないとき) を見てください。

### 9. 動かない方策で、評価を最後まで回す

```bash
bash scripts/09_run_hold.sh
```

Genie Sim に付属の「動かない方策」（今の関節の位置をそのまま返すだけのサーバ）を、コンテナの中で立てます。そのうえで評価を実行し、観測、推論、行動、判定、録画の一周を最後まで回します。

- 既定のタスクのシーン 0 には、指示が 5 つあります。指示ごとに 1 エピソードを行うので、5 エピソードになります。
- ロボットは動かないので、どのエピソードも時間切れで終わります。**点数は 0 で正常です。** 確かめたいのは、仕組みが最後まで回ることです。
- 「採点結果が出ました」「録画が…本あります」と出れば成功です。録画（`*.webm`）は、ブラウザや VLC で開けます。

ここまで通れば、この PC で Genie Sim の評価の仕組みが動いています。

## 実行の記録の置き場所

`08_preview.sh`、`09_run_hold.sh`、`10_run_policy.sh` は、実行のたびに、記録を次の場所に作ります。`<種類>` は `preview`、`hold`、`policy` のどれかです。

```
~/genie_sim_ws/genie_sim/output/hs_runs/<種類>_<日時>/
```

| ファイル | 中身 |
|---|---|
| `run.log` | 実行の出力 |
| `gpu.csv` | 実行の間の GPU の使用量（2 秒ごと） |
| `preview_*.png` | カメラ画像（`08_preview.sh`） |
| `evaluate_ret_*.json` | 採点結果（`09_run_hold.sh` と `10_run_policy.sh`。タスクごとのディレクトリの下） |
| `*.webm` | 録画（`09_run_hold.sh` と `10_run_policy.sh`。頭と両手首のカメラ） |
| `policy_server.log` | 動かない方策のサーバの出力（`09_run_hold.sh`） |

## 画面ありで見る

Isaac Sim の画面を開いて、動いている様子を見ることもできます。

1. 画面のある端末（PC のデスクトップ）から、手順 7 を実行します。SSH 越しに起動したコンテナには画面がつながっていないので、一度止めて、起動し直します。
2. `.env` の `RUN_WINDOW` を `true` に変えます。
3. 手順 8 か 9 を実行します。

画面を開くと、その分の VRAM も使います。まずは画面なしで通してから試すのを勧めます。

## 2 台目の PC を使う

方策のサーバを、別の PC で動かす練習です。最初は、動かない方策のサーバをそのまま使います。

### 2 台目の PC（方策を動かす側）

Docker も GPU も要りません。このリポジトリを取得して、次の 2 つを実行します。

```bash
bash scripts/04_fetch.sh
```

```bash
bash scripts/hold_server.sh
```

サーバが起動し、この PC のアドレスを表示します。止めるときは Ctrl+C です。認証のないサーバなので、信頼できるネットワークの中だけで使います。

### 1 台目の PC（Genie Sim を動かす側）

`.env` の `INFER_HOST` を、2 台目のアドレスに書き換えます。

```
INFER_HOST=192.168.1.20:8999
```

そのうえで、次を実行します。

```bash
bash scripts/10_run_policy.sh
```

手順 9 と同じ結果が、ネットワーク越しに出れば成功です。つながらないときは、2 台目のファイアウォールで、ポート 8999 が閉じていないかを確かめます。

応答の遅い方策をまねることもできます。2 台目で、次のように起動します。1 回の応答を 0.5 秒遅らせます。

```bash
DELAY=0.5 bash scripts/hold_server.sh
```

### 自分の方策をつなぐ

`10_run_policy.sh` は、`INFER_HOST` で待ち受けている方策サーバなら、何でもつなげます。方策サーバは、WebSocket の上で msgpack をやりとりします。約束ごとは、資料の「方策との約束」にまとめています。Genie Sim の `scripts/fake_infer_server.py`（110 行）が、いちばん小さい実装例です。

学習済みの方策を使うなら、Genie Sim に π0.5 の方策サーバの立て方の案内があります（`source/geniesim_benchmark/skills/robocoliseum/challenge-baseline-model/SKILL.md`）。推論に使う VRAM は 8 GB 未満とされているので、2 台目の RTX 3090 で動かせる見込みです。こちらは確認できていません。

## 設定（.env）

| 項目 | 既定 | 意味 |
|---|---|---|
| `GS_WORKDIR` | `$HOME/genie_sim_ws` | 取得したものを置く場所 |
| `GENIESIM_REPO_URL` | `https://github.com/AgibotTech/genie_sim.git` | Genie Sim の取得元 |
| `GENIESIM_REF` | `083793f…` | Genie Sim の版（コミット） |
| `ASSET_SELECTORS` | `g2op_if` | 取得するアセットの範囲。空白で区切って複数書ける |
| `ASSET_SOURCE` | `huggingface` | アセットの取得元。`huggingface` か `modelscope` |
| `ASSET_WORKERS` | `8` | アセットを同時に取得する数 |
| `IMAGE_MODE` | `pull` | `pull` はビルド済みを取得、`build` は手元でビルド |
| `PREBUILT_IMAGE` | `yonhdee/genie_sim:geniesim4` | 取得するビルド済みイメージ |
| `GS_IMAGE` | `registry.agibot.com/genie-sim/geniesim4:latest` | Genie Sim が使うイメージの名前。変えなくてよい |
| `GS_CONTAINER` | `geniesim4` | コンテナの名前。変えなくてよい |
| `LIBUIPC_SM` | `89` | 手元でビルドするときの libuipc の対象。RTX 3090 では指定が要る |
| `CONTAINER_GUI` | `auto` | コンテナに画面をつなぐか。`auto` は、画面があればつなぐ |
| `UP_TIMEOUT` | `900` | コンテナの準備を待つ秒数の上限 |
| `SMOKE_TASK` | `g2op_if_pick_block_color` | 確認に使うタスク |
| `SMOKE_INSTANCE_IDS` | `0` | 確認に使うシーンの番号 |
| `RUN_WINDOW` | `false` | 実行のときに Isaac Sim の画面を開くか |
| `EXTRA_ARGS` | 空 | 評価のコマンドに足す設定。`--名前=値` の形で書く |
| `HOLD_PORT` | `8999` | 動かない方策のサーバのポート |
| `INFER_HOST` | `127.0.0.1:8999` | `10_run_policy.sh` がつなぐ方策サーバ |

タスクは、スクリプトの引数でも指定できます。アセットを取得した範囲のタスクを指定します。

```bash
bash scripts/08_preview.sh g2op_if_pick_block_shape
```

## うまくいかないとき

| 症状 | 原因と対処 |
|---|---|
| 手順 1 で「NG」が出る | 同じ行に直し方が出ます。直してから、もう一度実行します |
| `could not select device driver` と出て、コンテナが GPU を使えない | NVIDIA Container Toolkit が入っていないか、Docker に登録されていません。手順 3 を実行します |
| 手順 6 のビルドが `Unsupported host GPU capability '8.6'` で止まる | libuipc の対象が指定されていません。`.env` に `LIBUIPC_SM=89` があるかを確かめます |
| 手順 6 のビルドで、`nvcr.io` からの取得に認証を求められる | NVIDIA NGC の API キーを作り、`docker login nvcr.io` でログインします |
| 起動のログに「libuipc unavailable」と出る | RTX 3090 では想定どおりです。対処は要りません |
| 手順 7 で「アセットがありません」と出る | 手順 5 が終わっていません。もう一度実行します |
| 手順 7 で「コンテナが止まった状態で残っています」と出る | `bash scripts/down.sh` で片付けてから、手順 7 をやり直します |
| 手順 7 が、上限まで待っても終わらない | `docker logs -f geniesim4` で続きを見ます。進んでいるなら、`.env` の `UP_TIMEOUT` を延ばします |
| 手順 8 や 9 が、長いあいだ進まない | 初回は、描画の準備に時間がかかります。別の端末で `bash scripts/gpu_log.sh` を実行して、GPU が使われているかを見ます |
| 手順 8 や 9 が、数分たってから終了コード 1 で止まる | Genie Sim には、処理が 300 秒進まないと打ち切る仕組みがあります。もう一度実行します。キャッシュが残るので、2 回目は速くなります。それでも止まるときは、`.env` に `EXTRA_ARGS="--app.main_loop_stall_timeout_s=900"` と書きます |
| ログに Vulkan や RTX、ドライバについてのエラーが出る | ドライバの版が合っていない可能性があります。[ドライバについて](#ドライバについて) を見てください |
| VRAM が足りないというエラーが出る | GPU を使っている他のプログラムを止めます。画面を開いているなら、`RUN_WINDOW=false` に戻します |
| Python の `ImportError` や `ModuleNotFoundError` が出る | ビルド済みイメージの中身と、取得した版が合っていない可能性があります。`.env` の `IMAGE_MODE` を `build` に変えて、手順 6 からやり直します |
| 手順 9 で「ポート 8999 はすでに使われています」と出る | 前回の実行が残っています。`bash scripts/stop_run.sh` で止めます。別のプログラムが使っているなら、`.env` の `HOLD_PORT` を変えます |
| 画面ありにしたのに、画面が出ない | デスクトップの端末から手順 7 を実行したかを確かめます。Wayland で出ないときは、ログインの画面で「Ubuntu on Xorg」を選びます |
| 実行を途中で止めたい | Ctrl+C で止めます。止まらないときは、別の端末で `bash scripts/stop_run.sh` を実行します |

コンテナの中に入って調べることもできます。

```bash
bash scripts/into.sh
```

## ドライバについて

Isaac Sim 6.2 の動作確認は、Linux のドライバ 595.58.03 で行われています。今の版は、次のコマンドで分かります。

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

まずは、今のドライバのままで手順 8 まで試すのを勧めます。動けば、それで十分です。描画まわりのエラーが出るときに、更新を考えます。

入れられる版は、次のコマンドで分かります。

```bash
ubuntu-drivers list
```

ドライバを更新すると、ほかの環境（CUDA を使う学習の環境や、別のコンテナ）にも影響します。更新したあとは、PC を再起動します。

## 記録を残す

動いたかどうかにかかわらず、記録をまとめられます。

```bash
bash scripts/save_log.sh
```

PC の構成、版、コンテナのログ、直近 3 回の実行の記録を、`log/<日時>/` に集めます。録画は大きいので、既定では含めません。最後に、リポジトリに追加するためのコマンドを表示します。

このリポジトリは公開です。記録には、利用者名や PC の名前が入ることがあります。載せたくない情報がないかを確かめてから、プッシュしてください。

## 片付け

コンテナを止めます。

```bash
bash scripts/down.sh
```

取得したものを消します。本体、アセット、キャッシュ、実行の記録が、すべて消えます。

```bash
rm -rf ~/genie_sim_ws
```

Docker イメージを消します。

```bash
docker image rm registry.agibot.com/genie-sim/geniesim4:latest
```

```bash
docker image rm yonhdee/genie_sim:geniesim4
```

Isaac Sim のキャッシュが入ったボリュームを消します。

```bash
docker volume rm geniesim4-isaacsim-kit-cache
```

```bash
docker volume rm geniesim4-ovrtx-cache
```

## 中身

```
genie_sim_rtx3090/
├── README.md               この手順書
├── .env.example            設定のひな形
├── scripts/
│   ├── 01_check_host.sh            PC の状態を確かめる
│   ├── 02_install_docker.sh        Docker を入れる
│   ├── 03_install_nvidia_toolkit.sh  NVIDIA Container Toolkit を入れる
│   ├── 04_fetch.sh                 Genie Sim の本体を取得する
│   ├── 05_assets.sh                アセットを取得する
│   ├── 06_image.sh                 Docker イメージを用意する
│   ├── 07_up.sh                    コンテナを起動する
│   ├── 08_preview.sh               カメラ画像を 1 組保存する
│   ├── 09_run_hold.sh              動かない方策で評価を回す
│   ├── 10_run_policy.sh            自分の方策サーバをつないで評価する
│   ├── hold_server.sh              動かない方策のサーバを立てる（2 台目の PC 用）
│   ├── into.sh                     コンテナの中に入る
│   ├── down.sh                     コンテナを止める
│   ├── stop_run.sh                 残った実行を止める
│   ├── gpu_log.sh                  GPU の使用量を表示し続ける
│   ├── save_log.sh                 記録をまとめる
│   └── lib.sh                      共通の部品
└── log/                    save_log.sh がまとめた記録
```

取得したものは、既定では次のように置かれます。

```
~/genie_sim_ws/
├── genie_sim/          Genie Sim の本体
│   └── output/hs_runs/   実行の記録
├── assets/             アセット
├── venv/               geniesim コマンドの入った仮想環境
└── isaac-cache/        Isaac Sim のキャッシュ
```

## ライセンス

- Genie Sim のコードは MPL-2.0 です。
- アセットには、コードとは別のライセンスがあります。取得したあとの `~/genie_sim_ws/assets/LICENSE` を読んでください。非営利に限る条件（CC BY-NC-SA 4.0）だと伝えられていますが、こちらでは原文を確認できていません。**業務で使う前に、必ず確かめてください。**
- Isaac Sim と、ビルド済みのイメージにも、それぞれの利用条件があります。

## 次の一歩

- **タスクを増やす。** `.env` の `ASSET_SELECTORS` を広げて、手順 5 をもう一度実行します。
- **方策を入れ替える。** 動かない方策を、決まった動きを返す方策や、学習済みの方策に替えます。
- **対話型のシミュレーション（RT Engine）を動かす。** ROS 2 のノードとして動く別の仕組みです。アセットの全体（17.70 GiB）と、ROS 2 のワークスペースのビルドが要ります。この手順書では扱っていません。

進め方の全体は、資料の「進め方」にまとめています。

## 参考

- [Genie Sim（GitHub）](https://github.com/AgibotTech/genie_sim)
- [Isaac Sim 6.2 の動作要件](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
- [Docker Engine を Ubuntu に入れる](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit を入れる](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
