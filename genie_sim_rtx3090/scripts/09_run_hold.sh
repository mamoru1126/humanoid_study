#!/bin/bash
# 2つ目の動作確認. 「動かない方策」をつないで、評価の1周（観測、推論、行動、判定、録画）を最後まで回す.
#   bash scripts/09_run_hold.sh [タスク名]
#
# 動かない方策とは、Genie Sim に付属の scripts/fake_infer_server.py のこと.
# 受け取った関節角をそのまま返すので、ロボットは姿勢を保つ. 学習済みのモデルも、2台目の PC も要らない.
# ロボットが動かないので、点数は 0 になるのが正常. 確かめたいのは、仕組みが最後まで回ること.
# エピソードは、シーンにある指示の数だけ行う（既定のタスクのシーン 0 は指示が 5 つなので、5 エピソード）.
. "$(dirname "$0")/lib.sh"

require_container

task="${1:-${SMOKE_TASK}}"
check_name "${task}"
check_name "${SMOKE_INSTANCE_IDS}"
check_name "${HOLD_PORT}"
headless="$(resolve_headless)"

# コンテナはホストのネットワークを使うので、ホストでポートが空いているかを見ればよい
if (exec 3<> "/dev/tcp/127.0.0.1/${HOLD_PORT}") 2> /dev/null; then
  die "ポート ${HOLD_PORT} はすでに使われています. .env の HOLD_PORT を変えるか、使っているプロセスを止めてください（前回の残りなら bash scripts/stop_run.sh）."
fi

run="hold_$(stamp)"
host_dir="${RUNS_DIR}/${run}"
cont_dir="${RUNS_DIR_IN_CONTAINER}/${run}"
mkdir -p "${host_dir}"

cleanup() {
  gpu_log_stop
  stop_in_container "fake_infer_server.py"
}
trap cleanup EXIT

step "動かない方策のサーバを起動（127.0.0.1:${HOLD_PORT}）"
docker exec -d -u "$(id -u):$(id -g)" -e HOME=/home/isaac-sim -w /workspace "${GS_CONTAINER}" \
  bash -lc "exec python3 scripts/fake_infer_server.py --listen 127.0.0.1:${HOLD_PORT} > ${cont_dir}/policy_server.log 2>&1"
listening=0
for _ in $(seq 1 30); do
  if [ -f "${host_dir}/policy_server.log" ] && grep "listening on" "${host_dir}/policy_server.log" > /dev/null 2>&1; then
    listening=1
    break
  fi
  sleep 1
done
if [ "${listening}" != "1" ]; then
  cat "${host_dir}/policy_server.log" 2> /dev/null || true
  die "方策のサーバが起動しません. 上の出力を確かめてください."
fi
ok "起動しました."

gpu_log_start "${host_dir}/gpu.csv"

step "評価の実行（タスク: ${task}、シーン: ${SMOKE_INSTANCE_IDS}、画面: $([ "${headless}" = "true" ] && echo なし || echo あり)）"
extra="$(extra_args)"
cmd="set -o pipefail; geniesim benchmark run ${task}"
cmd+=" --infer-host=127.0.0.1:${HOLD_PORT}"
cmd+=" --benchmark.instance_ids=${SMOKE_INSTANCE_IDS}"
cmd+=" --benchmark.num_episode=1"
cmd+=" --benchmark.record=true"
cmd+=" --benchmark.output_dir=${cont_dir}"
cmd+=" --app.headless=${headless}"
cmd+="${extra}"
cmd+=" 2>&1 | tee ${cont_dir}/run.log"
say "コンテナの中で実行するコマンド:"
say "  ${cmd}"
say "指示ごとに1エピソードを行います（既定のタスクのシーン 0 は指示が 5 つ）. 動かない方策では、どのエピソードも時間切れで終わります."
say ""

status=0
in_container "${cmd}" || status=$?
gpu_log_stop

step "結果"
gpu_log_peak "${host_dir}/gpu.csv"
say "記録の置き場所: ${host_dir}"
n_json="$(find "${host_dir}" -name 'evaluate_ret_*.json' | wc -l)"
n_video="$(find "${host_dir}" -name '*.webm' | wc -l)"
if [ "${n_json}" -ge 1 ]; then
  ok "採点結果が出ました."
  find "${host_dir}" -name 'evaluate_ret_*.json' | sort
else
  ng "採点結果（evaluate_ret_*.json）がありません（終了コード: ${status}）."
fi
if [ "${n_video}" -ge 1 ]; then
  ok "録画が ${n_video} 本あります（ブラウザや VLC で開ける）."
  find "${host_dir}" -name '*.webm' | sort
else
  note "録画（*.webm）がありません."
fi
say "方策のサーバの記録: ${host_dir}/policy_server.log"
if [ "${n_json}" -ge 1 ]; then
  say ""
  say "ロボットは動かないので、点数は 0 で正常です."
  say "ここまで通れば、この PC で Genie Sim の評価の仕組みが動いています."
  say "自分の方策をつなぐときは bash scripts/10_run_policy.sh を使います."
  exit 0
fi
say "ログ: ${host_dir}/run.log"
say "bash scripts/save_log.sh で記録をまとめられます."
exit 1
