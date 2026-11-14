#!/bin/bash
# コンテナを止めて削除する.
#   bash scripts/down.sh
# 消えるのはコンテナだけ. 本体、アセット、キャッシュ、実行の記録、イメージは残る.
# 次に bash scripts/07_up.sh で起動し直すと、パッケージの登録からやり直す（数十秒から数分）.
. "$(dirname "$0")/lib.sh"

need_cmd docker "bash scripts/02_install_docker.sh を実行してください."
geniesim_cli docker down
