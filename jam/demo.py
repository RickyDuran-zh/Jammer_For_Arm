"""演示入口：默认实机（子板2 发模拟信号），--mock 为无硬件演示。"""
from __future__ import annotations

import random
import time
from dataclasses import replace

from .agent.brain import JamAgent
from .config import JamConfig
from .controller import JamController
from .radio.dual import MockDualRadio, make_radio


def _print_event(ev) -> None:
    if ev.kind == "acquired":
        print(f"  [锁定] frame={ev.frame} 信道{ev.channel} score={ev.score:.2f}")
    elif ev.kind == "hop":
        print(f"  [跟随] frame={ev.frame} 信道{ev.prev_channel} → 信道{ev.channel} "
              f"score={ev.score:.2f}")
    elif ev.kind == "lost":
        print(f"  [释放] frame={ev.frame} 原信道{ev.prev_channel}")


def victim_plan(cfg: JamConfig, seed: int | None = None) -> list[tuple[float, float]]:
    """子板2 的模拟通信信号：随机间隔（4~8 s）跳到随机信道。

    干扰机只通过感知拿数据，拿不到这份计划，所以跟随一定存在检测延迟。
    """
    rng = random.Random(seed)
    c = cfg.channel_centers_hz
    plan = [(0.0, c[2])]
    t, cur = 0.0, 3
    for _ in range(40):
        t += rng.uniform(4.0, 8.0)
        nxt = rng.choice([k for k in range(1, 11) if k != cur])
        cur = nxt
        plan.append((t, c[nxt - 1]))
    return plan


def _build(mock: bool, no_tx: bool, center_mhz: float):
    cfg = replace(JamConfig(), center_freq_hz=center_mhz * 1e6)
    if mock:
        radio = MockDualRadio(cfg, hop_plan=victim_plan(cfg),
                              victim_freq_hz=cfg.channel_centers_hz[2])
    else:
        radio = make_radio("uhd", cfg, tx_enabled=not no_tx)
        if not no_tx:
            radio.start_victim(victim_plan(cfg))
    return cfg, radio


def run_ui(mock: bool = False, no_tx: bool = False,
           center_mhz: float = 3000.0, fixed_ch: int | None = None) -> int:
    from PyQt5 import QtWidgets
    from .ui.app import MainWindow

    cfg, radio = _build(mock, no_tx, center_mhz)
    ctrl = JamController(cfg, radio, on_event=_print_event)
    ctrl.start()
    if fixed_ch:
        ctrl.start_fixed(fixed_ch)

    print("=" * 64)
    print(f"干扰机启动 | {'模拟后端' if mock else 'USRP X310'}"
          f" | 子板1: RX2 感知 + TX/RX 干扰 | 子板2: 模拟信号（随机 4~8 s 跳频）"
          f" | 中心 {center_mhz:.3f} MHz")
    print("界面窗口已打开，关掉窗口即退出。")
    print("=" * 64, flush=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(cfg, ctrl)
    win.show()
    win.raise_()
    win.activateWindow()
    try:
        return app.exec_()
    finally:
        ctrl.stop()
        radio.close()


def run_console(mock: bool = False, no_tx: bool = False,
                seconds: float = 10.0, center_mhz: float = 3000.0,
                chat: bool = False, fixed_ch: int | None = None) -> None:
    cfg, radio = _build(mock, no_tx, center_mhz)
    ctrl = JamController(cfg, radio, on_event=_print_event)
    ctrl.start()
    if fixed_ch:
        ctrl.start_fixed(fixed_ch)

    print(f"干扰机启动 | 中心 {center_mhz:.3f} MHz | "
          f"{'只感知' if no_tx else '感知+干扰'} | 按 Ctrl+C 结束")

    try:
        if chat:
            agent = JamAgent(ctrl)
            while True:
                text = input("操作员> ").strip()
                if text.lower() in ("exit", "quit", "q"):
                    break
                if text:
                    print(" Agent>", agent.handle(text))
        else:
            t0 = time.monotonic()
            while seconds <= 0 or time.monotonic() - t0 < seconds:
                time.sleep(0.5)
                st = ctrl.status()
                print(f"t={time.monotonic() - t0:4.1f}s 锁定={st.get('target_ch') or '-'} "
                      f"跳频={st.get('hop_count')} 中心={st.get('center_mhz'):.3f}MHz "
                      f"周期={st.get('cycle_ms')}ms TX={st.get('tx_frames')}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()
        radio.close()
