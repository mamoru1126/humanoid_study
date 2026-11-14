#!/bin/bash
# 共有するための記録を log/<日時>/ にまとめる. このあと git でコミットしてプッシュする.
#   bash scripts/save_log.sh                  直近 3 回の実行をまとめる
#   RUNS=5 bash scripts/save_log.sh           直近 5 回
#   WITH_VIDEO=1 bash scripts/save_log.sh     録画（*.webm）も含める（大きくなる）
#
# 入るもの: PC の状態、各ソフトの版、コンテナのログの末尾、実行ごとのログの末尾、
#           GPU の記録、採点結果、プレビュー画像、Isaac Sim（Kit）のログの末尾.
# リポジトリは公開なので、載せたくない情報（利用者名や機器名など）がないかを確かめてからプッシュする.
. "$(dirname "$0")/lib.sh"

out="${ENV_DIR}/log/$(stamp)"
mkdir -p "${out}"

step "PC の状態"
{
  echo "# $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "## uname"
  uname -srm
  echo
  echo "## nvidia-smi"
  nvidia-smi 2>&1 || true
  echo
  echo "## 01_check_host.sh"
  bash "${ENV_DIR}/scripts/01_check_host.sh" 2>&1 || true
} > "${out}/host.txt"
ok "host.txt"

step "各ソフトの版"
{
  echo "GENIESIM_REF（.env）: ${GENIESIM_REF}"
  if [ -d "${GENIESIM_DIR}/.git" ]; then
    echo "Genie Sim の本体: $(git -C "${GENIESIM_DIR}" rev-parse HEAD 2> /dev/null || echo 不明)"
    echo "Genie Sim の VERSION: $(tr -d '[:space:]' < "${GENIESIM_DIR}/VERSION" 2> /dev/null || echo 不明)"
  else
    echo "Genie Sim の本体: 未取得"
  fi
  echo "ASSET_SELECTORS: ${ASSET_SELECTORS}（取得元: ${ASSET_SOURCE}）"
  echo "IMAGE_MODE: ${IMAGE_MODE}"
  if command -v docker > /dev/null 2>&1; then
    echo "docker: $(docker --version 2>&1 || true)"
    echo "イメージ: $(docker image inspect "${GS_IMAGE}" --format '{{.Id}} 作成 {{.Created}} 大きさ {{.Size}}' 2> /dev/null || echo なし)"
    echo "コンテナ: $(container_state)"
  fi
  if command -v nvidia-ctk > /dev/null 2>&1; then
    echo "nvidia-ctk: $(nvidia-ctk --version 2>&1 | head -n 1)"
  fi
  echo "python3: $(python3 --version 2>&1 || true)"
} > "${out}/versions.txt"
ok "versions.txt"

step "コンテナのログ"
if [ -n "$(container_state)" ]; then
  docker logs --tail 300 "${GS_CONTAINER}" > "${out}/container.log" 2>&1 || true
  ok "container.log（末尾 300 行）"
else
  say "コンテナがないので飛ばします."
fi

step "実行の記録"
n_runs="${RUNS:-3}"
if [ -d "${RUNS_DIR}" ]; then
  mapfile -t runs < <(find "${RUNS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -rn | head -n "${n_runs}" | cut -d ' ' -f 2-)
else
  runs=()
fi
if [ "${#runs[@]}" -eq 0 ]; then
  say "実行の記録がありません（08〜10 をまだ実行していない）."
fi
for r in "${runs[@]}"; do
  name="$(basename "${r}")"
  dst="${out}/runs/${name}"
  mkdir -p "${dst}"
  if [ -f "${r}/run.log" ]; then
    tail -n 400 "${r}/run.log" > "${dst}/run_tail.log"
  fi
  if [ -f "${r}/policy_server.log" ]; then
    tail -n 100 "${r}/policy_server.log" > "${dst}/policy_server_tail.log"
  fi
  if [ -f "${r}/gpu.csv" ]; then
    cp "${r}/gpu.csv" "${dst}/gpu.csv"
  fi
  find "${r}" -name 'evaluate_ret_*.json' -exec cp {} "${dst}/" \;
  find "${r}" -maxdepth 1 -name 'preview_*.png' | sort | head -n 3 | while read -r f; do
    cp "${f}" "${dst}/"
  done
  if [ "${WITH_VIDEO:-0}" = "1" ]; then
    find "${r}" -name '*.webm' -exec cp {} "${dst}/" \;
  fi
  ok "runs/${name}"
done

step "Isaac Sim（Kit）のログ"
kit_log="$(find "${GS_CACHE_ROOT}/logs" "${GENIESIM_DIR}/.kit" -maxdepth 8 -type f -name '*.log' -printf '%T@ %p\n' 2> /dev/null | sort -rn | head -n 1 | cut -d ' ' -f 2- || true)"
if [ -n "${kit_log}" ]; then
  {
    echo "# ${kit_log}"
    tail -n 300 "${kit_log}"
  } > "${out}/kit_tail.log"
  ok "kit_tail.log（末尾 300 行）"
else
  say "Kit のログは見つかりませんでした."
fi

step "まとめ"
du -sh "${out}" | cut -f 1
say "保存先: ${out}"
say ""
say "内容を確かめてから、リポジトリの一番上のディレクトリで次を実行して共有します."
say "  git add genie_sim_rtx3090/log"
say "  git commit -m \"Genie Sim の実行記録を追加\""
say "  git push"
