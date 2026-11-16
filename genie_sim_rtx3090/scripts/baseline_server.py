"""学習済みの π0.5（大会のベースライン）を、この PC で方策サーバとして立てる.

  bash scripts/12_run_baseline.sh から呼ばれる（単体でも動く）.
  <ACoT-VLA の場所>/.venv/bin/python scripts/baseline_server.py --acot-dir <ACoT-VLA の場所> --port 8000

ベースラインの公式の起動方法（tunnel.sh）は、大会のサーバへ逆向きにつなぐもので、手元の Genie Sim とは
直接つながらない. ここでは、その中の推論の部分（tunnel_agent.PolicyHandler）だけを借りて、
Genie Sim 付属の fake_infer_server.py と同じ形（WebSocket で msgpack を 1 往復）で待ち受ける.
PolicyHandler は、Genie Sim が送る観測の形を、モデルの入力の形に変え、答えを Genie Sim の形に戻す.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acot-dir", required=True, help="ACoT-VLA を取得した場所")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default="pi05_genie_sim_instruction_and_robust_20260526")
    ap.add_argument("--ckpt-dir", default="checkpoints/instruction_and_robust_pi05")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    os.chdir(args.acot_dir)
    sys.path.insert(0, os.path.join(args.acot_dir, "scripts"))
    import websockets  # noqa: E402
    from tunnel_agent import PolicyHandler  # noqa: E402

    handler = PolicyHandler(args.config, args.ckpt_dir)
    t0 = time.monotonic()
    handler.load()
    logging.info("モデルを読み込みました（%.0f 秒）", time.monotonic() - t0)

    lock = asyncio.Lock()   # GPU の上のモデルは 1 つなので、推論は 1 本ずつ行う
    count = 0

    async def serve_conn(ws):
        nonlocal count
        logging.info("接続されました: %s", ws.remote_address)
        try:
            async for message in ws:
                async with lock:
                    t = time.monotonic()
                    reply = await asyncio.to_thread(handler, "local", message)
                    count += 1
                    if count <= 3 or count % 50 == 0:
                        logging.info("推論 %d 回目（%.0f ms）", count, (time.monotonic() - t) * 1000)
                await ws.send(reply)
        except websockets.exceptions.ConnectionClosed:
            pass
        logging.info("切断されました")

    async def run():
        async with websockets.serve(serve_conn, args.host, args.port, compression=None, max_size=None):
            print(f"listening on {args.host}:{args.port}", flush=True)
            await asyncio.Future()

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
