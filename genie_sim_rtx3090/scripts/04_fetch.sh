#!/bin/bash
# Genie Sim の本体を取得して、操作用のコマンド（geniesim）を入れる.
#   bash scripts/04_fetch.sh
#
# すること:
#   1. Genie Sim を GS_WORKDIR/genie_sim に取得する（.env の GENIESIM_REF の版）
#   2. ロボットのモデル（サブモジュール）を HTTPS で取得する
#   3. Python の仮想環境を GS_WORKDIR/venv に作り、geniesim コマンドを入れる
# 何度実行してもよい. ホストに入れるのは軽い CLI だけで、Isaac Sim などはコンテナの中にある.
. "$(dirname "$0")/lib.sh"

need_cmd git "sudo apt install git"
need_cmd python3 "sudo apt install python3 python3-venv"

SUBMODULE_PATH="source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model"

step "1. Genie Sim の本体（${GENIESIM_REF}）"
mkdir -p "${GS_WORKDIR}"
if [ ! -d "${GENIESIM_DIR}/.git" ]; then
  git init -q "${GENIESIM_DIR}"
  git -C "${GENIESIM_DIR}" remote add origin "${GENIESIM_REPO_URL}"
fi
current="$(git -C "${GENIESIM_DIR}" rev-parse -q --verify HEAD 2> /dev/null || true)"
if [ -n "${current}" ] && [ "${current}" = "${GENIESIM_REF}" ]; then
  ok "取得済みです: ${current}"
else
  git -C "${GENIESIM_DIR}" fetch --depth 1 origin "${GENIESIM_REF}"
  git -C "${GENIESIM_DIR}" checkout -q --detach FETCH_HEAD
  ok "取得しました: $(git -C "${GENIESIM_DIR}" rev-parse HEAD)"
fi
if [ -f "${GENIESIM_DIR}/VERSION" ]; then
  say "Genie Sim の版: $(tr -d '[:space:]' < "${GENIESIM_DIR}/VERSION")"
fi

step "2. ロボットのモデル（サブモジュール）"
# .gitmodules の URL は SSH 形式（git@github.com:...）で、GitHub に SSH 鍵を登録していないと取得に失敗する.
# 公開リポジトリなので、この作業用の取得先だけ HTTPS に書き換える（.gitmodules は変更しない）.
sub_name="$(git -C "${GENIESIM_DIR}" config -f .gitmodules --get-regexp '^submodule\..*\.path$' \
  | awk -v p="${SUBMODULE_PATH}" '$2 == p { print $1 }' \
  | sed -e 's/^submodule\.//' -e 's/\.path$//' || true)"
if [ -z "${sub_name}" ]; then
  note "サブモジュール ${SUBMODULE_PATH} が .gitmodules にありません（版によって構成が違う可能性）. 飛ばします."
else
  sub_url="$(git -C "${GENIESIM_DIR}" config -f .gitmodules --get "submodule.${sub_name}.url")"
  case "${sub_url}" in
    git@github.com:*) sub_url="https://github.com/${sub_url#git@github.com:}" ;;
  esac
  git -C "${GENIESIM_DIR}" submodule init -- "${SUBMODULE_PATH}"
  git -C "${GENIESIM_DIR}" config "submodule.${sub_name}.url" "${sub_url}"
  git -C "${GENIESIM_DIR}" submodule update -- "${SUBMODULE_PATH}"
  ok "取得しました: ${sub_url}"
fi

step "3. geniesim コマンド"
if [ ! -x "${GS_VENV}/bin/python" ]; then
  if ! python3 -m venv "${GS_VENV}"; then
    die "仮想環境を作れません. sudo apt install python3-venv を実行してからやり直してください."
  fi
fi
"${GS_VENV}/bin/python" -m pip install --quiet --upgrade pip
"${GS_VENV}/bin/python" -m pip install --quiet -e "${GENIESIM_DIR}/source/geniesim_cli"
if [ ! -x "${GS_VENV}/bin/geniesim" ]; then
  die "geniesim コマンドが入りませんでした. 上のエラーを確かめてください."
fi
geniesim_cli version || true
ok "geniesim コマンドを入れました: ${GS_VENV}/bin/geniesim"

say ""
say "次は bash scripts/05_assets.sh です."
