#!/bin/bash
# 最初の動作確認. シーンを読み込み、ロボットの3つのカメラの画像を1組保存して終わる.
#   bash scripts/08_preview.sh [タスク名]
#
# 方策（policy）サーバは要らない. Isaac Sim の起動、アセットの読み込み、描画までを確かめられる.
# 初回はシェーダのコンパイルがあるので、数分から十数分かかることがある.
. "$(dirname "$0")/lib.sh"

require_container

task="${1:-${SMOKE_TASK}}"
check_name "${task}"
check_name "${SMOKE_INSTANCE_IDS}"
headless="$(resolve_headless)"

run="preview_$(stamp)"
host_dir="${RUNS_DIR}/${run}"
cont_dir="${RUNS_DIR_IN_CONTAINER}/${run}"
mkdir -p "${host_dir}"

cleanup() {
  gpu_log_stop
}
trap cleanup EXIT
gpu_log_start "${host_dir}/gpu.csv"

step "プレビューの実行（タスク: ${task}、シーン: ${SMOKE_INSTANCE_IDS}、画面: $([ "${headless}" = "true" ] && echo なし || echo あり)）"
extra="$(extra_args)"
cmd="set -o pipefail; geniesim benchmark run ${task}"
cmd+=" --benchmark.preview=true"
cmd+=" --benchmark.instance_ids=${SMOKE_INSTANCE_IDS}"
cmd+=" --benchmark.record=false"
cmd+=" --app.headless=${headless}"
cmd+="${extra}"
cmd+=" 2>&1 | tee ${cont_dir}/run.log"
say "コンテナの中で実行するコマンド:"
say "  ${cmd}"
say ""

status=0
in_container "${cmd}" "GENIESIM_PREVIEW_OUTPUT_DIR=${cont_dir}" || status=$?
gpu_log_stop

step "結果"
n_png="$(find "${host_dir}" -maxdepth 1 -name 'preview_*.png' | wc -l)"
gpu_log_peak "${host_dir}/gpu.csv"
say "記録の置き場所: ${host_dir}"
if [ "${n_png}" -ge 3 ]; then
  ok "カメラ画像を ${n_png} 枚保存しました（頭、左手首、右手首）."
  find "${host_dir}" -maxdepth 1 -name 'preview_*.png' | sort
  say ""
  say "画像を開いて、机と物体とロボットの腕が写っていることを確かめてください."
  say "次は bash scripts/09_run_hold.sh です."
  exit 0
fi
ng "カメラ画像が保存されていません（終了コード: ${status}）."
say "ログ: ${host_dir}/run.log"
say "原因の手がかりは README の「うまくいかないとき」にあります. bash scripts/save_log.sh で記録をまとめられます."
exit 1
