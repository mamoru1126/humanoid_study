#!/bin/bash
# GPU のメモリ使用量、使用率、温度を、2 秒おきに表示する（止めるときは Ctrl+C）.
#   bash scripts/gpu_log.sh
# 別の端末で動かしておくと、起動や実行の間に VRAM がどこまで増えるかを見られる.
# 08〜10 のスクリプトは同じ内容を自動で記録する（記録の置き場所の gpu.csv）.
. "$(dirname "$0")/lib.sh"

need_cmd nvidia-smi "NVIDIA のドライバを入れてください."
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv -l 2
