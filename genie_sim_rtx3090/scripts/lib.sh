#!/bin/bash
# 共通の設定と関数. 各スクリプトが最初に読み込む（単体では実行しない）.

set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "${ENV_DIR}/.env" ]; then
  cp "${ENV_DIR}/.env.example" "${ENV_DIR}/.env"
fi
set -a
# shellcheck disable=SC1091
. "${ENV_DIR}/.env"
set +a

# ---- 既定値（.env に書かれていない項目のため）-----------------------------------
GS_WORKDIR="${GS_WORKDIR:-${HOME}/genie_sim_ws}"
GENIESIM_REPO_URL="${GENIESIM_REPO_URL:-https://github.com/AgibotTech/genie_sim.git}"
GENIESIM_REF="${GENIESIM_REF:-083793f238762d336ce8570d22118a60a6830875}"
ASSET_SELECTORS="${ASSET_SELECTORS:-g2op_if}"
ASSET_SOURCE="${ASSET_SOURCE:-huggingface}"
ASSET_WORKERS="${ASSET_WORKERS:-8}"
IMAGE_MODE="${IMAGE_MODE:-pull}"
PREBUILT_IMAGE="${PREBUILT_IMAGE:-yonhdee/genie_sim:geniesim4}"
GS_IMAGE="${GS_IMAGE:-registry.agibot.com/genie-sim/geniesim4:latest}"
GS_CONTAINER="${GS_CONTAINER:-geniesim4}"
LIBUIPC_SM="${LIBUIPC_SM:-89}"
CONTAINER_GUI="${CONTAINER_GUI:-auto}"
UP_TIMEOUT="${UP_TIMEOUT:-900}"
SMOKE_TASK="${SMOKE_TASK:-g2op_if_pick_block_color}"
SMOKE_INSTANCE_IDS="${SMOKE_INSTANCE_IDS:-0}"
RUN_WINDOW="${RUN_WINDOW:-false}"
HOLD_PORT="${HOLD_PORT:-8999}"
INFER_HOST="${INFER_HOST:-127.0.0.1:8999}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---- 置き場所（GS_WORKDIR の下）--------------------------------------------------
GENIESIM_DIR="${GS_WORKDIR}/genie_sim"
GS_VENV="${GS_WORKDIR}/venv"
GS_ASSETS_DIR="${GS_WORKDIR}/assets"
GS_CACHE_ROOT="${GS_WORKDIR}/isaac-cache"
# 実行の記録. コンテナからは /workspace/output/hs_runs に見える
RUNS_DIR="${GENIESIM_DIR}/output/hs_runs"
RUNS_DIR_IN_CONTAINER="/workspace/output/hs_runs"

# Isaac Sim 6.2 の動作確認に使われた Linux のドライバ（公式の要件のページの値）
TESTED_DRIVER="595.58.03"

# ---- 表示 ----------------------------------------------------------------------
say()  { printf '%s\n' "$*"; }
ok()   { printf 'OK:   %s\n' "$*"; }
note() { printf '注意: %s\n' "$*"; }
ng()   { printf 'NG:   %s\n' "$*" >&2; }
die()  { ng "$*"; exit 1; }
step() { printf '\n== %s\n' "$*"; }

need_cmd() {
  if ! command -v "$1" > /dev/null 2>&1; then
    die "$1 が見つかりません. $2"
  fi
}

confirm() {
  if [ "${ASSUME_YES:-0}" = "1" ]; then
    return 0
  fi
  local reply=""
  read -r -p "$1 [y/N]: " reply || true
  case "${reply}" in
    y | Y | yes | YES) return 0 ;;
    *) return 1 ;;
  esac
}

# $1 >= $2 なら真（バージョン番号の比較）
ver_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

# 名前に使ってよい文字だけか（タスク名などをコマンドに埋め込む前に確かめる）
check_name() {
  if ! [[ "$1" =~ ^[A-Za-z0-9_.,:-]+$ ]]; then
    die "使えない文字が入っています: $1"
  fi
}

stamp() { date +%Y%m%d_%H%M%S; }

# 評価のコマンドに足す設定（.env の EXTRA_ARGS）を、先頭に空白を付けて返す. 「--名前=値」の形だけを通す.
extra_args() {
  local token out=""
  for token in ${EXTRA_ARGS}; do
    if ! [[ "${token}" =~ ^--[A-Za-z0-9_.-]+=[A-Za-z0-9_.,:/-]+$ ]]; then
      die "EXTRA_ARGS は「--名前=値」を空白で区切って書いてください. 通らなかった部分: ${token}"
    fi
    out+=" ${token}"
  done
  printf '%s' "${out}"
}

# ---- Genie Sim の CLI（ホスト側）--------------------------------------------------
geniesim_cli() {
  if [ ! -x "${GS_VENV}/bin/geniesim" ]; then
    die "geniesim コマンドがありません. 先に bash scripts/04_fetch.sh を実行してください."
  fi
  GENIESIM_REPO_ROOT="${GENIESIM_DIR}" \
  GENIESIM_WORKSPACE="${GENIESIM_DIR}" \
  GENIESIM_ASSETS_SRC="${GS_ASSETS_DIR}" \
  GENIESIM_CACHE_ROOT="${GS_CACHE_ROOT}" \
  GENIESIM_IMAGE="${GS_IMAGE}" \
  GENIESIM_CONTAINER="${GS_CONTAINER}" \
    "${GS_VENV}/bin/geniesim" "$@"
}

# ---- コンテナ --------------------------------------------------------------------
container_state() {
  docker inspect -f '{{.State.Status}}' "${GS_CONTAINER}" 2> /dev/null || true
}

require_container() {
  need_cmd docker "bash scripts/02_install_docker.sh を実行してください."
  local state
  state="$(container_state)"
  if [ "${state}" != "running" ]; then
    die "コンテナ ${GS_CONTAINER} が動いていません（状態: ${state:-なし}）. bash scripts/07_up.sh を実行してください."
  fi
}

# コンテナに画面（X11）がつながっているか
container_has_display() {
  local mounts
  mounts="$(docker inspect -f '{{range .Mounts}}{{println .Destination}}{{end}}' "${GS_CONTAINER}" 2> /dev/null || true)"
  case $'\n'"${mounts}"$'\n' in
    *$'\n'/tmp/.X11-unix$'\n'*) return 0 ;;
    *) return 1 ;;
  esac
}

# コンテナの中でコマンドを実行する. geniesim docker into と同じ条件（利用者、HOME、ログインシェル）にする.
#   in_container "<コマンド>" [名前=値 ...]     2つ目以降は、コンテナに渡す環境変数
# 端末から実行したときは端末をつなぐので、Ctrl+C がコンテナの中のプロセスに届く.
in_container() {
  local cmd="$1"
  shift
  local args=(-u "$(id -u):$(id -g)" -e HOME=/home/isaac-sim -w /workspace)
  local kv
  for kv in "$@"; do
    args+=(-e "${kv}")
  done
  if [ -t 0 ] && [ -t 1 ]; then
    args+=(-it)
  else
    args+=(-i)
  fi
  docker exec "${args[@]}" "${GS_CONTAINER}" bash -lc "${cmd}"
}

# 端末をつながずに実行する（出力を変数に取るとき、後片付けのとき）
in_container_quiet() {
  docker exec -u "$(id -u):$(id -g)" -e HOME=/home/isaac-sim -w /workspace \
    "${GS_CONTAINER}" bash -lc "$1"
}

# コンテナの中で、コマンドラインに $1 を含むプロセスへ SIGINT を送る（pkill に頼らない）
stop_in_container() {
  local pattern="$1"
  docker exec -u "$(id -u):$(id -g)" "${GS_CONTAINER}" bash -c '
    pattern="$1"
    for d in /proc/[0-9]*; do
      pid="${d#/proc/}"
      if [ "${pid}" = "$$" ]; then
        continue
      fi
      if tr "\0" " " < "${d}/cmdline" 2> /dev/null | grep -q -- "${pattern}"; then
        kill -INT "${pid}" 2> /dev/null || true
      fi
    done
  ' _ "${pattern}" > /dev/null 2>&1 || true
}

# 実行のときに Isaac Sim の画面を開くかを決める. 結果は true / false（--app.headless に渡す値）
resolve_headless() {
  if [ "${RUN_WINDOW}" = "true" ]; then
    if container_has_display; then
      echo "false"
      return 0
    fi
    note "RUN_WINDOW=true ですが、このコンテナには画面がつながっていません. 画面なしで実行します." >&2
  fi
  echo "true"
}

# ---- GPU の記録（ホスト側の nvidia-smi）---------------------------------------------
GPU_LOG_PID=""

gpu_log_start() {
  local file="$1"
  if ! command -v nvidia-smi > /dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,nounits -l 2 > "${file}" 2> /dev/null &
  GPU_LOG_PID="$!"
}

gpu_log_stop() {
  if [ -n "${GPU_LOG_PID}" ]; then
    kill "${GPU_LOG_PID}" 2> /dev/null || true
    wait "${GPU_LOG_PID}" 2> /dev/null || true
    GPU_LOG_PID=""
  fi
}

# 記録から、GPU 0 のメモリ使用量の最大値を表示する
gpu_log_peak() {
  local file="$1"
  if [ ! -s "${file}" ]; then
    return 0
  fi
  awk -F', ' '
    NR > 1 && $2 == 0 { if ($3 + 0 > peak) peak = $3 + 0; total = $4 + 0 }
    END { if (total > 0) printf "GPU 0 のメモリ使用量の最大: %d MiB / %d MiB（この実行の間. 他のプロセスの分も含む）\n", peak, total }
  ' "${file}"
}
