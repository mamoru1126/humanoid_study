#!/bin/bash
# NVIDIA Container Toolkit を入れて、Docker のコンテナから GPU を使えるようにする（NVIDIA 公式の apt の手順）.
# すでに使える状態なら、確認だけする. sudo を使う.
#   bash scripts/03_install_nvidia_toolkit.sh
#
# 版を固定したいときは、環境変数で指定する（例: NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1）.
. "$(dirname "$0")/lib.sh"

need_cmd docker "先に bash scripts/02_install_docker.sh を実行してください."
need_cmd nvidia-smi "先に NVIDIA のドライバを入れてください."

# グループ docker が有効になる前でも進められるように、必要なら sudo をつける
DOCKER=(docker)
if ! docker info > /dev/null 2>&1; then
  DOCKER=(sudo docker)
fi

has_nvidia_runtime() {
  local runtimes
  runtimes="$("${DOCKER[@]}" info --format '{{json .Runtimes}}' 2> /dev/null || true)"
  [[ "${runtimes}" == *nvidia* ]]
}

if has_nvidia_runtime; then
  ok "Docker に NVIDIA のランタイムが登録されています. インストールは飛ばします."
else
  say "NVIDIA Container Toolkit を入れます."
  say "行うこと: 鍵とリポジトリの登録、パッケージのインストール、Docker の設定の書き換え（/etc/docker/daemon.json）、Docker の再起動."
  say "Docker を再起動するので、動いているコンテナはいったん止まります."
  if ! confirm "続けますか"; then
    exit 1
  fi

  step "鍵とリポジトリの登録"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
  keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  if [ ! -f "${keyring}" ]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o "${keyring}"
  fi
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=${keyring}] https://#g" \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

  step "インストール"
  sudo apt-get update
  v="${NVIDIA_CONTAINER_TOOLKIT_VERSION:-}"
  if [ -n "${v}" ]; then
    sudo apt-get install -y \
      "nvidia-container-toolkit=${v}" \
      "nvidia-container-toolkit-base=${v}" \
      "libnvidia-container-tools=${v}" \
      "libnvidia-container1=${v}"
  else
    sudo apt-get install -y nvidia-container-toolkit
  fi

  step "Docker の設定"
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

step "確認（コンテナの中で nvidia-smi を実行する）"
"${DOCKER[@]}" run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
ok "コンテナから GPU が見えました."
say "次は bash scripts/04_fetch.sh です."
