"""本振独立性自检。

目的：确认把 TX 本振调到目标信道时，RX（感知）本振是否被一起带走。

X310 的两块 UBX-160 有的配置 RX/TX 共用本振，有的独立。共用的话，
"TX 跟着信道走"会让感知窗口一起平移，必须改用基带搬移方案。

用法（设备必须空闲，先关掉正在跑的界面）：
    python run.py --probe-lo
"""
from __future__ import annotations

from .config import JamConfig


def probe_lo(cfg: JamConfig | None = None) -> int:
    from .radio.dual import UhdDualRadio

    cfg = cfg or JamConfig()
    centers = cfg.centers_at(cfg.center_freq_hz)
    print("=" * 68)
    print("本振独立性自检：RX 固定在感知中心，TX 依次调到 1/6/10 号信道")
    print("=" * 68)
    try:
        radio = UhdDualRadio(cfg, tx_enabled=True)
    except Exception as exc:
        print(f"打开设备失败：{exc}")
        print("（如果界面正在运行，会占用 X310，请先关掉再试）")
        return 1

    shared = False
    try:
        rx0 = radio.rx_freq()
        print(f"初始：RX {rx0 / 1e6:.5f} MHz ｜ TX {radio.tx_freq() / 1e6:.5f} MHz")
        for ch in (1, 6, 10):
            want = float(centers[ch - 1])
            radio._tune_tx(want)
            rx = radio.rx_freq()
            tx = radio.tx_freq()
            moved = abs(rx - rx0) > 1e3
            shared = shared or moved
            print(f"信道{ch:>2}：目标 {want / 1e6:.5f} MHz → "
                  f"实测 TX {tx / 1e6:.5f} / RX {rx / 1e6:.5f} MHz ｜ "
                  f"调谐 {radio.last_tune_ms:.1f} ms ｜ RX{'被带走' if moved else '未动'}")
    finally:
        try:
            radio.close()
        except Exception:
            pass

    print("-" * 68)
    if shared:
        print("结论：RX/TX 共用本振 → 不能用 TX 跟踪模式，请加 --no-tx-track 用基带搬移。")
        return 2
    print("结论：RX/TX 本振独立 → TX 可以直接调到目标信道，基带打 DC。")
    return 0
