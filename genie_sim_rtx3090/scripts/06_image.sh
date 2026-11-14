#!/bin/bash
# Genie Sim の Docker イメージ（Isaac Sim 6.2、ROS 2 Jazzy、依存パッケージ入り）を用意する.
#   bash scripts/06_image.sh
#
# .env の IMAGE_MODE で方法を選ぶ.
#   pull  : 公開されているビルド済みイメージを取得して、CLI が使う名前をつける（既定）
#   build : 手元でビルドする. RTX 3090 では LIBUIPC_SM の指定が要る（下の説明を参照）
# すでにイメージがあれば何もしない. 取り直すときは FORCE=1 bash scripts/06_image.sh
. "$(dirname "$0")/lib.sh"

need_cmd docker "bash scripts/02_install_docker.sh を実行してください."
if ! docker info > /dev/null 2>&1; then
  die "docker を使えません. bash scripts/01_check_host.sh で状態を確かめてください."
fi

show_image() {
  docker image inspect "${GS_IMAGE}" --format 'イメージ: {{index .RepoTags 0}}  大きさ: {{.Size}} バイト  作成: {{.Created}}'
}

if [ "${FORCE:-0}" != "1" ] && docker image inspect "${GS_IMAGE}" > /dev/null 2>&1; then
  ok "イメージはすでにあります."
  show_image
  say "次は bash scripts/07_up.sh です."
  exit 0
fi

case "${IMAGE_MODE}" in
  pull)
    step "ビルド済みイメージの取得（${PREBUILT_IMAGE}）"
    say "大きなイメージです. 回線によっては時間がかかります. 途中で切れたら、もう一度実行すると続きから取得します."
    docker pull "${PREBUILT_IMAGE}"
    docker tag "${PREBUILT_IMAGE}" "${GS_IMAGE}"
    ;;
  build)
    step "手元でのビルド"
    # Genie Sim の CLI は、ビルドの前に GPU 0 の計算能力（compute capability）を調べ、
    # libuipc が対応する 8.9（RTX 40 系）か 12.0（RTX 50 系）でなければ止まる.
    # RTX 3090 は 8.6 なので、--build-arg LIBUIPC_SM=89 を明示して進める.
    # ビルドは GPU を使わないので通る. 3090 では libuipc を使うタスクだけが動かない.
    case "${LIBUIPC_SM}" in
      89 | 120) ;;
      *) die ".env の LIBUIPC_SM は 89 か 120 にしてください（今の値: ${LIBUIPC_SM}）." ;;
    esac
    if [ ! -d "${GENIESIM_DIR}/docker" ]; then
      die "Genie Sim の本体がありません. 先に bash scripts/04_fetch.sh を実行してください."
    fi
    say "1 時間以上かかることがあります. 元になる Isaac Sim のイメージは nvcr.io から取得します."
    GENIESIM_LIBUIPC_SOURCE="${ASSET_SOURCE}" \
      geniesim_cli docker build --build-arg "LIBUIPC_SM=${LIBUIPC_SM}"
    ;;
  *)
    die ".env の IMAGE_MODE は pull か build にしてください（今の値: ${IMAGE_MODE}）."
    ;;
esac

if ! docker image inspect "${GS_IMAGE}" > /dev/null 2>&1; then
  die "イメージ ${GS_IMAGE} ができていません. 上の出力を確かめてください."
fi
ok "イメージを用意しました."
show_image
say ""
say "次は bash scripts/07_up.sh です."
