#!/bin/bash
# アセット（シーン、物体、ロボットのモデル、背景画像）を、使うタスクの分だけ取得する.
#   bash scripts/05_assets.sh
#
# 取得する範囲は .env の ASSET_SELECTORS で決める. 途中で止めても、もう一度実行すれば続きから再開する.
# 取得先は GS_WORKDIR/assets. 起動すると、コンテナの /geniesim_assets につながる.
. "$(dirname "$0")/lib.sh"

# shellcheck disable=SC2206
selectors=(${ASSET_SELECTORS})
if [ "${#selectors[@]}" -eq 0 ]; then
  die ".env の ASSET_SELECTORS が空です."
fi
for s in "${selectors[@]}"; do
  check_name "${s}"
done
case "${ASSET_SOURCE}" in
  huggingface | modelscope) ;;
  *) die ".env の ASSET_SOURCE は huggingface か modelscope にしてください（今の値: ${ASSET_SOURCE}）." ;;
esac

step "取得する範囲（${selectors[*]}）"
geniesim_cli assets plan "${selectors[@]}" --source "${ASSET_SOURCE}"

step "ダウンロード（${ASSET_SOURCE}）"
mkdir -p "${GS_ASSETS_DIR}"
geniesim_cli assets download "${selectors[@]}" \
  --source "${ASSET_SOURCE}" \
  --dest "${GS_ASSETS_DIR}" \
  --workers "${ASSET_WORKERS}"

step "検査（ファイルの欠けと、中身の一致）"
geniesim_cli assets verify "${selectors[@]}" --source "${ASSET_SOURCE}" --dest "${GS_ASSETS_DIR}"

if [ ! -f "${GS_ASSETS_DIR}/pyproject.toml" ]; then
  die "${GS_ASSETS_DIR}/pyproject.toml がありません. コンテナの起動に必要です. ダウンロードの出力を確かめてください."
fi
ok "アセットを取得しました: ${GS_ASSETS_DIR}（$(du -sh "${GS_ASSETS_DIR}" | cut -f 1)）"
if [ -f "${GS_ASSETS_DIR}/LICENSE" ]; then
  say "アセットの利用条件は ${GS_ASSETS_DIR}/LICENSE にあります. 業務で使う前に読んでください."
fi

say ""
say "次は bash scripts/06_image.sh です."
