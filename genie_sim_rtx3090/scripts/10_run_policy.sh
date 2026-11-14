#!/bin/bash
# 自分で用意した方策（policy）サーバをつないで評価する.
#   bash scripts/10_run_policy.sh [タスク名]
#
# つなぐ先は .env の INFER_HOST（例: 192.168.1.20:8999）. 別の PC で動かしていてよい.
# 方策サーバは、WebSocket の上で msgpack をやりとりする（約束ごとは資料の「方策との約束」を参照）.
# エピソードは、シーンにある指示の数だけ行う（既定のタスクのシーン 0 は指示が 5 つなので、5 エピソード）.
# 先に疎通を確かめたいときは CHECK=1 bash scripts/10_run_policy.sh
. "$(dirname "$0")/lib.sh"

require_container

task="${1:-${SMOKE_TASK}}"
check_name "${task}"
check_name "${SMOKE_INSTANCE_IDS}"
check_name "${INFER_HOST}"
headless="$(resolve_headless)"

host="${INFER_HOST%:*}"
port="${INFER_HOST##*:}"
if ! (exec 3<> "/dev/tcp/${host}/${port}") 2> /dev/null; then
  die "${INFER_HOST} に接続できません. 方策サーバが動いているか、.env の INFER_HOST が正しいか、ファイアウォールでポートが閉じていないかを確かめてください."
fi
ok "${INFER_HOST} に接続できます."

if [ "${CHECK:-0}" = "1" ]; then
  step "方策サーバとの疎通の確認（Genie Sim 付属の検査）"
  say "付属の検査は、決まった形の観測を1回送って応答を調べます. ロボットの種類が合わないサーバでは失敗することがあります."
  in_container "geniesim benchmark check-inference --infer-host=${INFER_HOST}" || note "検査が失敗しました. 上の出力を確かめてください."
fi

run="policy_$(stamp)"
host_dir="${RUNS_DIR}/${run}"
cont_dir="${RUNS_DIR_IN_CONTAINER}/${run}"
mkdir -p "${host_dir}"

cleanup() {
  gpu_log_stop
}
trap cleanup EXIT
gpu_log_start "${host_dir}/gpu.csv"

step "評価の実行（タスク: ${task}、シーン: ${SMOKE_INSTANCE_IDS}、方策: ${INFER_HOST}）"
extra="$(extra_args)"
cmd="set -o pipefail; geniesim benchmark run ${task}"
cmd+=" --infer-host=${INFER_HOST}"
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
if [ "${status}" != "0" ]; then
  ng "実行が終了コード ${status} で終わりました. ログ: ${host_dir}/run.log"
  exit 1
fi
ok "終わりました."
