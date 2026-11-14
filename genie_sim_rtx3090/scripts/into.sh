#!/bin/bash
# 動いているコンテナに入る（抜けるときは exit）.
#   bash scripts/into.sh
# 中では /workspace が Genie Sim の本体、/geniesim_assets がアセット. geniesim コマンドがそのまま使える.
. "$(dirname "$0")/lib.sh"

require_container
geniesim_cli docker into
