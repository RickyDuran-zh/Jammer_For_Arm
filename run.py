"""入口。

    python run.py            实机界面：子板1 感知+干扰，子板2 发模拟信号
    python run.py --mock     无硬件演示
    python run.py --no-tx    实机只感知，不发射
    python run.py --console  不开界面，命令行看状态
    python run.py --selftest 离线自检
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jam import selftest
from jam.demo import run_console, run_ui


def main() -> int:
    ap = argparse.ArgumentParser(description="感知-跟踪-干扰一体机")
    ap.add_argument("--mock", action="store_true", help="无硬件演示")
    ap.add_argument("--no-tx", action="store_true", help="只感知，不发射干扰")
    ap.add_argument("--console", action="store_true", help="不开界面，命令行运行")
    ap.add_argument("--seconds", type=float, default=10.0, help="命令行模式运行时长")
    ap.add_argument("--center", type=float, default=3000.0, help="中心频率 MHz")
    ap.add_argument("--fixed", type=int, default=None,
                    help="固定干扰某个信道（1-10），不自动跟随")
    ap.add_argument("--chat", action="store_true", help="命令行模式下用自然语言控制")
    ap.add_argument("--selftest", action="store_true", help="离线自检")
    args = ap.parse_args()

    if args.selftest:
        return selftest.main()
    if args.console:
        run_console(args.mock, args.no_tx, args.seconds, args.center, args.chat,
                    args.fixed)
        return 0
    return run_ui(args.mock, args.no_tx, args.center, args.fixed)


if __name__ == "__main__":
    raise SystemExit(main())
