"""入口。

    python run.py            实机界面：子板1 感知+干扰（子板2 默认关闭）
    python run.py --mock     无硬件演示
    python run.py --no-tx    实机只感知，不发射
    python run.py --victim   启动时打开子板2模拟信号
    python run.py --console  不开界面，命令行看状态
    python run.py --selftest 离线自检

三种干扰模式（对应申报书表 2）：
    固定干扰    --jam-mode fixed  --fixed 6
    响应式干扰  --jam-mode track                  （默认，锁定后跟随切频）
    随机干扰    --jam-mode random --random-period 2

联调常用：
    python run.py --jam-mode fixed  --fixed 6 --waveform multitone --bw 320 --power 15 --log case01.jsonl
    python run.py --jam-mode random --random-period 2 --log case02.jsonl
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
    ap.add_argument("--jam-mode", choices=["fixed", "track", "random"],
                    default="track",
                    help="干扰模式：fixed=固定 / track=响应式 / random=随机")
    ap.add_argument("--fixed", type=int, default=None,
                    help="固定干扰某个信道（1-10），隐含 --jam-mode fixed")
    ap.add_argument("--random-period", type=float, default=2.0,
                    help="随机干扰的参数变化周期 秒（默认 2.0）")
    ap.add_argument("--victim", action="store_true", default=None,
                    help="启动时打开子板2模拟信号（默认只在 --mock 下打开）")
    ap.add_argument("--waveform", choices=["noise", "multitone"], default=None,
                    help="干扰波形：noise=限带噪声 multitone=多音")
    ap.add_argument("--bw", type=float, default=None,
                    help="干扰带宽 kHz（160/320/480/640/1600）")
    ap.add_argument("--power", type=float, default=None,
                    help="干扰功率 dB（-30~20，负值用于减弱）")
    ap.add_argument("--react-delay-ms", type=float, default=None,
                    help="响应式跟踪的反应延迟 ms：判定切频后等这么久才搬干扰（0=立即）")
    ap.add_argument("--span", type=int, default=None,
                    help="相邻信道覆盖数 1-10：一次压住连续 N 个信道（覆盖 --bw）")
    ap.add_argument("--jam-ms", type=float, default=None,
                    help="一帧干扰时长 ms（默认 400；短帧 20 适合快速跟踪）")
    ap.add_argument("--no-park-tx", action="store_true",
                    help="待机时不要把 TX 本振挪出频段（默认会挪，避免泄漏压在 3 GHz 上）")
    ap.add_argument("--lo-hop", action="store_true",
                    help="TDD 本振跳变：发射前整块搬到目标信道、发完搬回"
                         "（RX/TX 共用本振时用，消除 3.000000 常驻泄漏）")
    ap.add_argument("--log", type=str, default=None,
                    help="把状态与事件追加记录为 JSONL，便于联调对比")
    ap.add_argument("--case", type=str, default="",
                    help="本次用例名，写入记录首行，便于区分")
    ap.add_argument("--tx-gain", type=float, default=None,
                    help="TX 射频增益 dB（UBX 范围 0-31.5，默认取配置 12；要更强就往上加）")
    ap.add_argument("--chat", action="store_true", help="命令行模式下用自然语言控制")
    ap.add_argument("--no-tx-track", action="store_true",
                    help="TX 本振固定：仍用基带搬移对准信道（默认 TX 跟着信道走）")
    ap.add_argument("--probe-lo", action="store_true",
                    help="自检 RX/TX 本振是否独立（设备需空闲）")
    ap.add_argument("--selftest", action="store_true", help="离线自检")
    args = ap.parse_args()

    if args.selftest:
        return selftest.main()

    if args.probe_lo:
        from jam.probe import probe_lo
        return probe_lo()

    params = {}
    if args.waveform is not None:
        params["waveform_mode"] = 1 if args.waveform == "multitone" else 0
    if args.bw is not None:
        params["bw_khz"] = float(args.bw)
    if args.power is not None:
        params["power_db"] = float(args.power)
    if args.react_delay_ms is not None:
        params["react_delay_ms"] = float(args.react_delay_ms)
    if args.span is not None:
        if not 1 <= args.span <= 10:
            ap.error("--span 必须在 1-10 之间")
        params["span_channels"] = int(args.span)
    if args.jam_ms is not None:
        if not 2 <= args.jam_ms <= 5000:
            ap.error("--jam-ms 必须在 2-5000 之间")
        params["jam_ms"] = float(args.jam_ms)
    if args.lo_hop:
        params["lo_hop"] = True
    case = args.case or (Path(args.log).stem if args.log else "")

    jam_mode = "fixed" if args.fixed is not None else args.jam_mode
    if jam_mode == "fixed" and args.fixed is None:
        args.fixed = 6          # 没指定信道就默认压 3.000000 GHz
    if jam_mode != "fixed":
        args.fixed = None

    common = dict(victim=args.victim, params=params or None,
                  log_path=args.log, case=case, tx_gain_db=args.tx_gain,
                  jam_mode=jam_mode, random_period=args.random_period,
                  tx_lo_track=not args.no_tx_track,
                  span_channels=(args.span or 1), park_tx=not args.no_park_tx)
    if args.console:
        run_console(args.mock, args.no_tx, args.seconds, args.center, args.chat,
                    args.fixed, **common)
        return 0
    return run_ui(args.mock, args.no_tx, args.center, args.fixed, **common)


if __name__ == "__main__":
    raise SystemExit(main())
