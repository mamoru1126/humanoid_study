#!/bin/bash
# この PC で Genie Sim を動かせるかを調べる. 何も変更しない.
#   bash scripts/01_check_host.sh
. "$(dirname "$0")/lib.sh"

n_ng=0
n_note=0
bad()  { ng "$*"; n_ng=$((n_ng + 1)); }
warn() { note "$*"; n_note=$((n_note + 1)); }

step "OS"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:22.04 | ubuntu:24.04) ok "${PRETTY_NAME}" ;;
    *) warn "${PRETTY_NAME:-不明} です. 手順は Ubuntu 22.04 と 24.04 を想定しています." ;;
  esac
else
  warn "/etc/os-release が読めません."
fi
if [ "$(uname -m)" != "x86_64" ]; then
  bad "CPU が x86_64 ではありません（$(uname -m)）."
fi

step "GPU"
if ! command -v nvidia-smi > /dev/null 2>&1; then
  bad "nvidia-smi が見つかりません. NVIDIA のドライバを入れてください."
else
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
  gpu_name="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader 2> /dev/null || true)"
  gpu_mem="$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits 2> /dev/null || true)"
  gpu_drv="$(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader 2> /dev/null || true)"
  gpu_cap="$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader 2> /dev/null || true)"

  if [ -z "${gpu_name}" ]; then
    bad "GPU 0 の情報が取れません. nvidia-smi の出力を確かめてください."
  else
    case "${gpu_name}" in
      *RTX*) ok "GPU 0: ${gpu_name}（RTX. レイトレーシング用の回路がある）" ;;
      *) warn "GPU 0: ${gpu_name}. Isaac Sim は RT コアのある GPU が必要です（A100、H100 などは対象外）." ;;
    esac
  fi

  if [[ "${gpu_mem}" =~ ^[0-9]+$ ]]; then
    if [ "${gpu_mem}" -ge 16000 ]; then
      ok "VRAM: ${gpu_mem} MiB（Isaac Sim 6.2 の最小要件は 16 GB）"
    else
      bad "VRAM: ${gpu_mem} MiB. Isaac Sim 6.2 の最小要件の 16 GB に足りません."
    fi
  fi

  if [[ "${gpu_drv}" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
    if ver_ge "${gpu_drv}" "${TESTED_DRIVER}"; then
      ok "ドライバ: ${gpu_drv}（Isaac Sim 6.2 の動作確認は ${TESTED_DRIVER}）"
    else
      warn "ドライバ: ${gpu_drv}. Isaac Sim 6.2 の動作確認は ${TESTED_DRIVER} です. 古いと起動や描画で失敗することがあります（README の「ドライバについて」を参照）."
    fi
  fi

  case "${gpu_cap}" in
    8.9 | 12.0)
      ok "計算能力（compute capability）: ${gpu_cap}. libuipc も使えます."
      ;;
    "")
      warn "計算能力を取得できません（nvidia-smi が古い可能性）."
      ;;
    *)
      say "情報: 計算能力は ${gpu_cap}. libuipc（布などの変形物体のソルバ）は 8.9 と 12.0 だけに対応しています."
      say "      RTX 3090 は 8.6 なので、libuipc を使う3タスクは動きません. ほかのタスクには影響しません."
      say "      イメージを手元でビルドするときは LIBUIPC_SM の指定が要ります（.env に設定済み）."
      ;;
  esac
fi

step "メモリとディスク"
mem_gb="$(awk '/MemTotal/ { printf "%d", $2 / 1024 / 1024 }' /proc/meminfo)"
if [ "${mem_gb}" -ge 30 ]; then
  ok "メモリ: ${mem_gb} GB（Isaac Sim 6.2 の最小要件は 32 GB）"
else
  warn "メモリ: ${mem_gb} GB. Isaac Sim 6.2 の最小要件は 32 GB です."
fi

free_gb() {
  local p="$1"
  while [ ! -e "${p}" ] && [ "${p}" != "/" ]; do
    p="$(dirname "${p}")"
  done
  df -BG --output=avail "${p}" | tail -n 1 | tr -dc '0-9'
}
work_free="$(free_gb "${GS_WORKDIR}")"
if [ "${work_free}" -ge 30 ]; then
  ok "作業用の置き場所（${GS_WORKDIR}）の空き: ${work_free} GB"
else
  warn "作業用の置き場所（${GS_WORKDIR}）の空き: ${work_free} GB. 本体、アセット、キャッシュで 15 GB 以上使います."
fi

step "Docker"
docker_ok=0
if ! command -v docker > /dev/null 2>&1; then
  bad "docker が見つかりません. bash scripts/02_install_docker.sh を実行してください."
else
  ok "$(docker --version)"
  if docker info > /dev/null 2>&1; then
    docker_ok=1
    ok "sudo なしで docker を使えます."
    docker_root="$(docker info --format '{{.DockerRootDir}}' 2> /dev/null || echo /var/lib/docker)"
    docker_free="$(free_gb "${docker_root}")"
    if [ "${docker_free}" -ge 60 ]; then
      ok "Docker の置き場所（${docker_root}）の空き: ${docker_free} GB"
    else
      warn "Docker の置き場所（${docker_root}）の空き: ${docker_free} GB. イメージは数十 GB になる見込みです."
    fi
    runtimes="$(docker info --format '{{json .Runtimes}}' 2> /dev/null || true)"
    if [[ "${runtimes}" == *nvidia* ]]; then
      ok "Docker に NVIDIA のランタイムが登録されています."
    else
      bad "Docker に NVIDIA のランタイムがありません. bash scripts/03_install_nvidia_toolkit.sh を実行してください."
    fi
  else
    bad "docker を sudo なしで使えません（グループ docker に入っていないか、デーモンが止まっています）. bash scripts/02_install_docker.sh を実行してください."
  fi
fi

step "そのほかの道具"
if command -v git > /dev/null 2>&1; then
  ok "$(git --version)"
else
  bad "git が見つかりません. sudo apt install git"
fi
if command -v python3 > /dev/null 2>&1; then
  py_ver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  case "${py_ver}" in
    3.10 | 3.11 | 3.12) ok "python3 ${py_ver}" ;;
    *) bad "python3 ${py_ver}. Genie Sim の CLI は 3.10 から 3.12 が対象です." ;;
  esac
  if python3 -c 'import ensurepip' > /dev/null 2>&1; then
    ok "python3 の venv が使えます."
  else
    bad "python3 の venv が使えません. sudo apt install python3-venv"
  fi
else
  bad "python3 が見つかりません."
fi

step "画面"
if [ -n "${DISPLAY:-}" ]; then
  ok "DISPLAY=${DISPLAY}（${XDG_SESSION_TYPE:-種類は不明}）. 画面ありで起動できます."
  if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
    say "情報: Wayland です. 画面が出ないときは、ログイン画面で「Ubuntu on Xorg」を選ぶか、画面なしで使ってください."
  fi
else
  say "情報: DISPLAY がありません（SSH 越しなど）. 画面なしで起動します. 確認は、保存した画像と録画で行います."
fi

step "結果"
say "NG: ${n_ng} 件、注意: ${n_note} 件"
if [ "${n_ng}" -gt 0 ]; then
  say "NG の項目を直してから、もう一度実行してください."
  exit 1
fi
if [ "${docker_ok}" = "1" ]; then
  say "次は bash scripts/04_fetch.sh です（Docker と NVIDIA Container Toolkit は入っているので、02 と 03 は不要）."
fi
