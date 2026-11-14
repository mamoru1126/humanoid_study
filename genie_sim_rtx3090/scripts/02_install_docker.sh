#!/bin/bash
# Docker Engine を入れる（Docker 公式の apt リポジトリを使う方法）.
# すでに使える状態なら何もしない. sudo を使う.
#   bash scripts/02_install_docker.sh
. "$(dirname "$0")/lib.sh"

add_to_docker_group() {
  local me
  me="$(id -un)"
  case " $(id -nG "${me}") " in
    *" docker "*)
      say "利用者 ${me} はグループ docker に入っています. ログインし直すと有効になります."
      return 0
      ;;
  esac
  say "docker を sudo なしで使うには、利用者 ${me} をグループ docker に入れます."
  say "（グループ docker に入ると、root と同等の操作が docker 経由でできるようになります）"
  if confirm "グループ docker に追加しますか"; then
    sudo usermod -aG docker "${me}"
    say "追加しました. いったんログアウトしてログインし直すと有効になります."
    say "ログインし直したあと、bash scripts/01_check_host.sh で確かめてください."
  else
    say "追加しませんでした. Genie Sim のスクリプトは docker を sudo なしで呼ぶので、このままでは進めません."
  fi
}

if command -v docker > /dev/null 2>&1; then
  ok "Docker は入っています: $(docker --version)"
  if docker info > /dev/null 2>&1; then
    ok "sudo なしで使えます. このスクリプトですることはありません."
    exit 0
  fi
  if sudo docker info > /dev/null 2>&1; then
    add_to_docker_group
    exit 0
  fi
  die "Docker のデーモンに接続できません. sudo systemctl status docker で状態を確かめてください."
fi

if [ ! -r /etc/os-release ]; then
  die "/etc/os-release が読めません. Docker の公式手順（https://docs.docker.com/engine/install/）に従って入れてください."
fi
# shellcheck disable=SC1091
. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
  die "Ubuntu ではありません（${ID:-不明}）. Docker の公式手順（https://docs.docker.com/engine/install/）に従って入れてください."
fi

# 公式のパッケージとぶつかる、ディストリビューション側のパッケージが残っていないか
conflicts="$({ dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc 2> /dev/null || true; } | awk '$2 == "install" { print $1 }' | tr '\n' ' ')"
if [ -n "${conflicts}" ]; then
  ng "公式の Docker とぶつかるパッケージが入っています: ${conflicts}"
  say "公式の手順では、先にこれらを削除します. 内容を確かめたうえで、次を実行してからやり直してください."
  say "  sudo apt remove ${conflicts}"
  exit 1
fi

codename="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"
arch="$(dpkg --print-architecture)"

say "Docker Engine を、Docker 公式の apt リポジトリから入れます（${PRETTY_NAME}、${codename}、${arch}）."
say "行うこと: 鍵の登録、/etc/apt/sources.list.d/docker.sources の作成、パッケージのインストール."
if ! confirm "続けますか"; then
  exit 1
fi

step "鍵とリポジトリの登録"
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null << EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${codename}
Components: stable
Architectures: ${arch}
Signed-By: /etc/apt/keyrings/docker.asc
EOF

step "インストール"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

step "確認"
sudo docker run --rm hello-world
ok "Docker が動きました."

add_to_docker_group
say "次は bash scripts/03_install_nvidia_toolkit.sh です."
