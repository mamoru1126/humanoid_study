#!/bin/bash
# 学習済みの π0.5（AgiBot World Challenge のベースライン）を、この PC に用意する. 1回だけ実行すればよい.
#   bash scripts/11_setup_baseline.sh
#
# やること（どれもコンテナの外、この PC の上で行う）:
#   1. 推論のコード ACoT-VLA を取得する（GS_WORKDIR/acot_vla）
#   2. uv で Python の環境を作る（JAX と CUDA のライブラリも入る. 数 GB）
#   3. 重み instruction_and_robust_pi05 を ModelScope から取得する（大きい. 空きを十分に取る）
# 重みは、指示に従う課題（g2op_if_ で始まるタスク. 既定の g2op_if_pick_block_color を含む）向け.
. "$(dirname "$0")/lib.sh"

ACOT_DIR="${ACOT_DIR:-${GS_WORKDIR}/acot_vla}"
ACOT_REPO="${ACOT_REPO:-https://github.com/Anonymous-694/ACoT-VLA.git}"
ACOT_REF="${ACOT_REF:-9ded4eb4ce8fa3a969ba1d35a45528155381ea86}"   # agibot_world_challenge ブランチ（2026-11-16 時点）
CKPT_NAME="${CKPT_NAME:-instruction_and_robust_pi05}"

need_cmd git "sudo apt install git で入れてください."
need_cmd curl "sudo apt install curl で入れてください."

step "空き容量"
mkdir -p "${GS_WORKDIR}"
free_gb=$(df -BG --output=avail "${GS_WORKDIR}" | tail -1 | tr -dc '0-9')
say "${GS_WORKDIR} の空き: ${free_gb} GB"
if [ "${free_gb}" -lt 40 ]; then
  note "40 GB 未満です. Python の環境と重みで数十 GB 使うことがあります."
  confirm "このまま続けますか" || exit 1
fi

step "uv（Python の環境を作る道具）"
if ! command -v uv > /dev/null 2>&1 && [ -x "${HOME}/.local/bin/uv" ]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if command -v uv > /dev/null 2>&1; then
  ok "入っています（$(uv --version)）."
else
  say "公式の手順（https://docs.astral.sh/uv/）で、~/.local/bin に入れます."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  ok "入れました（$(uv --version)）."
fi

step "推論のコード（${ACOT_DIR}）"
if [ ! -d "${ACOT_DIR}/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "${ACOT_REPO}" "${ACOT_DIR}"
fi
git -C "${ACOT_DIR}" fetch --quiet origin agibot_world_challenge
GIT_LFS_SKIP_SMUDGE=1 git -C "${ACOT_DIR}" checkout --quiet "${ACOT_REF}"
ok "版: $(git -C "${ACOT_DIR}" rev-parse --short HEAD)"

step "Python の環境（初回は 10 分以上かかることがある）"
(
  cd "${ACOT_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
  uv pip install modelscope
)
ok "${ACOT_DIR}/.venv に作りました."

step "GPU が見えるか（JAX）"
if ! "${ACOT_DIR}/.venv/bin/python" -c 'import jax; d = jax.devices(); print(d); assert d[0].platform == "gpu"'; then
  die "JAX から GPU が見えません. ドライバが古いか、CUDA のライブラリが入っていない可能性があります. 上の出力を確かめてください."
fi
ok "見えます."

step "重み（${CKPT_NAME}）"
ckpt_dir="${ACOT_DIR}/checkpoints/${CKPT_NAME}"
if [ -d "${ckpt_dir}" ] && [ -n "$(ls -A "${ckpt_dir}" 2> /dev/null)" ]; then
  ok "取得済みです（${ckpt_dir}）. 取り直すときは、このディレクトリを消してから実行してください."
else
  (
    cd "${ACOT_DIR}"
    PATH="${ACOT_DIR}/.venv/bin:${PATH}" bash scripts/download_checkpoint.sh "${CKPT_NAME}" "${ACOT_DIR}/checkpoints"
  )
  ok "取得しました（$(du -sh "${ckpt_dir}" | cut -f1)）."
fi
if [ ! -d "${ckpt_dir}/assets" ]; then
  note "${ckpt_dir}/assets がありません. 正規化の統計がないと、推論の開始で止まります. 中身を確かめてください: ls ${ckpt_dir}"
fi

say ""
ok "準備ができました. 次は bash scripts/12_run_baseline.sh です（コンテナを起動したうえで）."
