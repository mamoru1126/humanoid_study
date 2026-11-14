#!/bin/bash
# コンテナの中に残った実行（評価のプロセスと、動かない方策のサーバ）を止める.
#   bash scripts/stop_run.sh
# 端末を閉じた、SSH が切れた、などで実行が残ったときに使う. コンテナは動かしたままにする.
. "$(dirname "$0")/lib.sh"

require_container
stop_in_container "geniesim_benchmark/app/app.py"
stop_in_container "fake_infer_server.py"
sleep 3
say "停止の合図（SIGINT）を送りました. コンテナの中で動いているプロセス:"
in_container_quiet "ps -eo pid,etime,cmd 2> /dev/null | grep -E 'app.py|fake_infer_server|geniesim' | grep -v grep" || say "（該当なし）"
say "まだ残っているときは、bash scripts/down.sh でコンテナごと止めてください."
