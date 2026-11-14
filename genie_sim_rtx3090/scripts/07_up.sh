#!/bin/bash
# Genie Sim のコンテナを起動して、準備ができるまで待つ.
#   bash scripts/07_up.sh
#
# 起動は Genie Sim の CLI（geniesim docker up）に任せる. CLI がすること:
#   - GPU をすべて渡し、ホストのネットワークを使い、特権つきでコンテナを起動する
#   - Genie Sim の本体を /workspace に、アセットを /geniesim_assets につなぐ（どちらも読み書きできる）
#   - コンテナの中の利用者を、ホストの利用者と同じ番号に合わせる
#   - 本体の Python パッケージを、コンテナの中に「編集可能」の形で入れる
# 画面ありのときは、CLI が xhost +local: を実行して、この PC の中からの X11 接続を許可する.
. "$(dirname "$0")/lib.sh"

need_cmd docker "bash scripts/02_install_docker.sh を実行してください."

if [ ! -d "${GENIESIM_DIR}/docker" ]; then
  die "Genie Sim の本体がありません. 先に bash scripts/04_fetch.sh を実行してください."
fi
if [ ! -f "${GS_ASSETS_DIR}/pyproject.toml" ]; then
  die "アセットがありません. 先に bash scripts/05_assets.sh を実行してください."
fi
if ! docker image inspect "${GS_IMAGE}" > /dev/null 2>&1; then
  die "イメージがありません. 先に bash scripts/06_image.sh を実行してください."
fi

state="$(container_state)"
if [ "${state}" = "running" ]; then
  ok "コンテナ ${GS_CONTAINER} はすでに動いています."
  say "次は bash scripts/08_preview.sh です. 設定を変えて起動し直すときは、先に bash scripts/down.sh を実行してください."
  exit 0
fi
if [ -n "${state}" ]; then
  die "コンテナ ${GS_CONTAINER} が止まった状態で残っています（${state}）. bash scripts/down.sh を実行してからやり直してください."
fi

# 画面をつなぐかどうか
gui="false"
case "${CONTAINER_GUI}" in
  true) gui="true" ;;
  false) gui="false" ;;
  auto)
    if [ -n "${DISPLAY:-}" ]; then
      gui="true"
    fi
    ;;
  *) die ".env の CONTAINER_GUI は auto、true、false のどれかにしてください（今の値: ${CONTAINER_GUI}）." ;;
esac

up_args=()
if [ "${gui}" = "true" ]; then
  if [ -z "${DISPLAY:-}" ]; then
    die "CONTAINER_GUI=true ですが、DISPLAY がありません. 画面のある端末から実行するか、CONTAINER_GUI=false にしてください."
  fi
  # CLI は ~/.Xauthority をコンテナにつなぐ. このファイルがないと、Docker が同じ名前の
  # ディレクトリを root の所有で作ってしまうので、先に空のファイルを作っておく.
  if [ -d "${HOME}/.Xauthority" ]; then
    die "${HOME}/.Xauthority がディレクトリになっています（以前の Docker の起動で作られた可能性）. 中身が空であることを確かめて削除してから、やり直してください."
  fi
  if [ ! -e "${HOME}/.Xauthority" ]; then
    touch "${HOME}/.Xauthority"
    say "${HOME}/.Xauthority がなかったので、空のファイルを作りました."
  fi
  say "画面ありで起動します（DISPLAY=${DISPLAY}）."
else
  up_args+=(--headless)
  say "画面なしで起動します."
fi

mkdir -p "${GS_CACHE_ROOT}" "${RUNS_DIR}"

step "コンテナの起動"
geniesim_cli docker up "${up_args[@]}"

step "準備ができるまで待つ（最大 ${UP_TIMEOUT} 秒. 初回はパッケージの登録に時間がかかる）"
waited=0
ready=0
while [ "${waited}" -lt "${UP_TIMEOUT}" ]; do
  if [ "$(container_state)" != "running" ]; then
    docker logs --tail 40 "${GS_CONTAINER}" 2>&1 || true
    die "コンテナが止まりました. 上のログを確かめてください."
  fi
  logs="$(docker logs "${GS_CONTAINER}" 2>&1 || true)"
  if [[ "${logs}" == *GENIESIM_READY* ]]; then
    ready=1
    break
  fi
  sleep 5
  waited=$((waited + 5))
done
if [ "${ready}" != "1" ]; then
  docker logs --tail 40 "${GS_CONTAINER}" 2>&1 || true
  die "${UP_TIMEOUT} 秒待っても準備が終わりません. 上のログを確かめてください（続きは docker logs -f ${GS_CONTAINER} で見られます）."
fi
ok "コンテナの準備ができました（${waited} 秒）."

if [[ "${logs,,}" == *libuipc* ]] && [[ "${logs,,}" == *unavailable* ]]; then
  say ""
  say "情報: ログに「libuipc unavailable」が出ています. RTX 3090 では想定どおりです."
  say "      libuipc（布などの変形物体のソルバ）は RTX 40 系と 50 系だけに対応しています."
  say "      動かないのは libuipc を使う3タスクだけで、ほかのタスクには影響しません."
fi

step "コンテナから見える GPU"
in_container_quiet "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader" || note "コンテナの中で nvidia-smi を実行できませんでした."

say ""
say "次は bash scripts/08_preview.sh です. コンテナに入るときは bash scripts/into.sh、止めるときは bash scripts/down.sh."
