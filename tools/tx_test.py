"""连续单音发射测试：不感知、不跟踪，直接在子板1 ch0 TX/RX 发 30 秒单音。

用法:
    python tools/tx_test.py --channel 5 --seconds 30
    python tools/tx_test.py --center 2400 --channel 3 --amp 0.25

频谱仪设置建议：中心 3 GHz、跨度 5 MHz、RBW 10 kHz、参考电平 -30 dBm。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jam.config import JamConfig


def main() -> int:
    ap = argparse.ArgumentParser(description="连续单音发射测试")
    ap.add_argument("--channel", type=int, default=5, help="信道 1-10")
    ap.add_argument("--center", type=float, default=3000.0, help="中心频率 MHz")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--amp", type=float, default=0.25, help="基带幅度 0-0.25")
    args = ap.parse_args()

    import uhd

    cfg = JamConfig()
    u = uhd.usrp.MultiUSRP(f"addr={cfg.ip_addr}")
    ch = 0
    u.set_tx_rate(cfg.sense_rate, ch)
    u.set_tx_freq(uhd.libpyuhd.types.tune_request(args.center * 1e6), ch)
    u.set_tx_gain(cfg.tx_gain_fixed, ch)
    u.set_tx_antenna(cfg.tx_antenna, ch)

    sa = uhd.usrp.StreamArgs("fc32", "sc16")
    sa.channels = [ch]
    tx = u.get_tx_stream(sa)

    centers = cfg.centers_at(args.center * 1e6)
    off = centers[min(max(args.channel, 1), 10) - 1] - args.center * 1e6
    freq = args.center * 1e6 + off
    n = 32768
    t = np.arange(n) / cfg.sense_rate
    blk = (min(max(args.amp, 0.02), 0.25)
           * np.exp(1j * 2 * np.pi * off * t)).astype(np.complex64).reshape(1, -1)

    md = uhd.types.TXMetadata()
    md.start_of_burst = False
    md.end_of_burst = False

    print(f"发射中：{freq/1e6:.4f} MHz（信道 {args.channel}），幅度 {args.amp}，"
          f"{args.seconds:.0f} 秒。Ctrl+C 停止。", flush=True)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            tx.send(blk, md, 1.0)
    except KeyboardInterrupt:
        pass
    finally:
        md.end_of_burst = True
        try:
            tx.send(np.zeros((1, 1), dtype=np.complex64), md, 1.0)
        except Exception:
            pass
        print("已停止发射", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
