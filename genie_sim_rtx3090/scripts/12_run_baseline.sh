#!/bin/bash
# 学習済みの π0.5 で、ロボットにタスクをやらせる（1台の PC で完結するデモ）.
#   bash scripts/12_run_baseline.sh [タスク名]
#
# この PC の上で π0.5 の方策サーバを立て（GPU を Isaac Sim と分け合う）、コンテナの中の Genie Sim からつなぐ.
# 先に bash scripts/11_setup_baseline.sh で準備し、bash scripts/07_up.sh でコンテナを起動しておく.
# 画面で見たいときは、.env を RUN_WINDOW=true にし、コンテナを画面ありで起動しておく.
# 既定のタスク g2op_if_pick_block_color は、重み instruction_and_robust_pi05 の学習に使われた課題の 1 つ.
. "$(dirname "$0")/lib.sh"

ACOT_DIR="${ACOT_DIR:-${GS_WORKDIR}/acot_vla}"
CKPT_NAME="${CKPT_NAME:-instruction_and_robust_pi05}"
BASELINE_CONFIG="${BASELINE_CONFIG:-pi05_genie_sim_instruction_and_robust_20260526}"
BASELINE_PORT="${BASELINE_PORT:-8000}"
# π0.5 に使わせる GPU のメモリの割合. Isaac Sim（画面ありで 6 GB ほど）と分け合うので、半分以下にする.
BASELINE_MEM_FRACTION="${BASELINE_MEM_FRACTION:-0.45}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-900}"

require_container

task="${1:-${SMOKE_TASK}}"
check_name "${task}"
check_name "${SMOKE_INSTANCE_IDS}"
check_name "${BASELINE_PORT}"
headless="$(resolve_headless)"

py="${ACOT_DIR}/.venv/bin/python"
if [ ! -x "${py}" ] || [ ! -d "${ACOT_DIR}/checkpoints/${CKPT_NAME}" ]; then
  die "π0.5 の準備ができていません. 先に bash scripts/11_setup_baseline.sh を実行してください."
fi
if (exec 3<> "/dev/tcp/127.0.0.1/${BASELINE_PORT}") 2> /dev/null; then
  die "ポート ${BASELINE_PORT} はすでに使われています. 前回のサーバが残っていないか確かめるか、BASELINE_PORT=8001 のように変えてください."
fi

run="baseline_$(stamp)"
host_dir="${RUNS_DIR}/${run}"
cont_dir="${RUNS_DIR_IN_CONTAINER}/${run}"
mkdir -p "${host_dir}"
server_log="${host_dir}/policy_server.log"

server_pid=""
cleanup() {
  gpu_log_stop
  if [ -n "${server_pid}" ] && kill -0 "${server_pid}" 2> /dev/null; then
    kill -INT "${server_pid}" 2> /dev/null || true
    sleep 2
    kill "${server_pid}" 2> /dev/null || true
  fi
}
trap cleanup EXIT
gpu_log_start "${host_dir}/gpu.csv"

step "π0.5 の方策サーバを起動（127.0.0.1:${BASELINE_PORT}. 重みの読み込みに数分かかる）"
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION="${BASELINE_MEM_FRACTION}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
XLA_FLAGS="--xla_gpu_autotune_level=0" \
TF_NUM_INTRAOP_THREADS=16 \
GIT_LFS_SKIP_SMUDGE=1 \
  "${py}" "${ENV_DIR}/scripts/baseline_server.py" \
    --acot-dir "${ACOT_DIR}" --port "${BASELINE_PORT}" \
    --config "${BASELINE_CONFIG}" --ckpt-dir "checkpoints/${CKPT_NAME}" \
    > "${server_log}" 2>&1 &
server_pid=$!
waited=0
while ! grep "listening on" "${server_log}" > /dev/null 2>&1; do
  if ! kill -0 "${server_pid}" 2> /dev/null; then
    tail -40 "${server_log}" >&2
    die "方策サーバが途中で止まりました. 上の出力と ${server_log} を確かめてください."
  fi
  if [ "${waited}" -ge "${LOAD_TIMEOUT}" ]; then
    tail -20 "${server_log}" >&2
    die "${LOAD_TIMEOUT} 秒たってもサーバの準備ができません. ${server_log} を確かめてください."
  fi
  sleep 5
  waited=$((waited + 5))
  if [ $((waited % 60)) -eq 0 ]; then
    say "  待っています（${waited} 秒）"
  fi
done
ok "起動しました（${waited} 秒）."

step "評価の実行（タスク: ${task}、シーン: ${SMOKE_INSTANCE_IDS}、画面: $([ "${headless}" = "true" ] && echo なし || echo あり)）"
say "指示ごとに 1 エピソードを行います（既定のタスクのシーン 0 は指示が 5 つ）. 最初の推論は、準備のために遅いことがあります."
extra="$(extra_args)"
cmd="set -o pipefail; geniesim benchmark run ${task}"
cmd+=" --infer-host=127.0.0.1:${BASELINE_PORT}"
cmd+=" --benchmark.instance_ids=${SMOKE_INSTANCE_IDS}"
cmd+=" --benchmark.num_episode=1"
cmd+=" --benchmark.record=true"
cmd+=" --benchmark.output_dir=${cont_dir}"
cmd+=" --app.headless=${headless}"
cmd+="${extra}"
cmd+=" 2>&1 | tee ${cont_dir}/run.log"
say "コンテナの中で実行するコマンド:"
say "  ${cmd}"
say ""

status=0
in_container "${cmd}" || status=$?
gpu_log_stop

step "結果"
gpu_log_peak "${host_dir}/gpu.csv"
say "記録の置き場所: ${host_dir}"
find "${host_dir}" -name 'evaluate_ret_*.json' | sort
find "${host_dir}" -name '*.webm' | sort
say "方策サーバの記録: ${server_log}"
if [ "${status}" != "0" ]; then
  ng "実行が終了コード ${status} で終わりました. ログ: ${host_dir}/run.log"
  exit 1
fi
ok "終わりました. 録画（*.webm）はブラウザや VLC で見られます."
