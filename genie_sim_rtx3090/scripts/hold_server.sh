#!/bin/bash
# 「動かない方策」のサーバを、この PC で立てる（止めるときは Ctrl+C）.
#   bash scripts/hold_server.sh
#
# 2台目の PC で使う. Genie Sim を動かす PC（1台目）から、ネットワーク越しに方策サーバへつなぐ練習になる.
# Docker も GPU も要らない. 要るのは Genie Sim の本体（04_fetch.sh で取得）と、Python のパッケージ2つ.
# 1台目では、.env の INFER_HOST をこの PC のアドレスにして、bash scripts/10_run_policy.sh を実行する.
#
# 応答を遅らせて、推論に時間のかかる方策をまねることもできる.
#   DELAY=0.5 bash scripts/hold_server.sh       1回の応答を 0.5 秒遅らせる
# 待ち受けるアドレスを絞りたいときは LISTEN_ADDR で指定する（既定は 0.0.0.0. すべてのアドレスで待ち受ける）.
#   LISTEN_ADDR=192.168.1.20 bash scripts/hold_server.sh
. "$(dirname "$0")/lib.sh"

server="${GENIESIM_DIR}/scripts/fake_infer_server.py"
if [ ! -f "${server}" ] || [ ! -x "${GS_VENV}/bin/python" ]; then
  die "Genie Sim の本体か、Python の仮想環境がありません. 先に bash scripts/04_fetch.sh を実行してください."
fi

listen_addr="${LISTEN_ADDR:-0.0.0.0}"
delay="${DELAY:-0}"
check_name "${HOLD_PORT}"
check_name "${listen_addr}"
if ! [[ "${delay}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  die "DELAY は秒数で指定してください（例: DELAY=0.5）. 今の値: ${delay}"
fi

if (exec 3<> "/dev/tcp/127.0.0.1/${HOLD_PORT}") 2> /dev/null; then
  die "ポート ${HOLD_PORT} はすでに使われています. .env の HOLD_PORT を変えるか、使っているプロセスを止めてください."
fi

step "Python のパッケージ（msgpack、websockets）"
if "${GS_VENV}/bin/python" -c 'import msgpack, websockets' > /dev/null 2>&1; then
  ok "入っています."
else
  "${GS_VENV}/bin/python" -m pip install --quiet msgpack 'websockets>=12'
  ok "入れました（${GS_VENV}）."
fi

step "動かない方策のサーバ（${listen_addr}:${HOLD_PORT}、応答の遅れ ${delay} 秒）"
say "この PC のアドレス: $(hostname -I 2> /dev/null || echo 不明)"
say "1台目の PC では、.env を INFER_HOST=<この PC のアドレス>:${HOLD_PORT} にして、bash scripts/10_run_policy.sh を実行します."
say "つながらないときは、この PC のファイアウォールでポート ${HOLD_PORT} が閉じていないかを確かめてください."
say "認証のないサーバです. 信頼できるネットワークの中だけで使ってください. 止めるときは Ctrl+C."
say ""
exec "${GS_VENV}/bin/python" "${server}" --listen "${listen_addr}:${HOLD_PORT}" --delay "${delay}"
