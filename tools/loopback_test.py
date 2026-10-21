"""端到端验证：子板2 发模拟信号（信道 3→7→2），子板1 感知并跟踪干扰。

用法: python tools/loopback_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jam.config import JamConfig
from jam.controller import JamController
from jam.radio.dual import UhdDualRadio


def main() -> int:
    cfg = JamConfig()
    c = cfg.channel_centers_hz
    radio = UhdDualRadio(cfg, tx_enabled=True, rx_thread=False)
    radio.start_victim([(0.0, c[2]), (3.0, c[6]), (6.0, c[1])])

    ctrl = JamController(cfg, radio, on_event=lambda e: print(
        f"  [{e.kind}] ch{e.prev_channel}->{e.channel} {e.detail}"))
    ctrl.start()

    print("阶段1：模拟信号在信道3")
    time.sleep(2.0)
    st = ctrl.status()
    print(f"  target={st['target_ch']} hops={st['hop_count']} "
          f"scores={[round(v, 2) for v in st['scores']]}")

    print("阶段2：信号切到信道7")
    time.sleep(2.5)
    st = ctrl.status()
    print(f"  target={st['target_ch']} hops={st['hop_count']}")

    print("阶段3：信号切到信道2")
    time.sleep(2.5)
    st = ctrl.status()
    print(f"  target={st['target_ch']} hops={st['hop_count']}")

    ctrl.stop()
    radio.close()

    ok = st.get("hop_count", 0) >= 2 and st.get("target_ch") == 2
    print("结果:", "PASS" if ok else "FAIL", st.get("hop_count"), "次跟随，"
          f"最终信道 {st.get('target_ch')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
