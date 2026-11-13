"""离线自检：跟踪状态机 + 感知 + 波形 + 闭环 + 中心频率搜索。"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .config import Action, JamConfig
from .controller import JamController
from .dsp.sensing import covered_channels, sense
from .dsp.waveform import generate
from .radio.dual import MockDualRadio
from .tracking.tracker import State, Tracker, TrackerPolicy

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def test_tracker() -> None:
    pol = TrackerPolicy(acquire_frames=2, hop_confirm_frames=2, release_frames=5,
                        on_threshold=0.55, off_threshold=0.30, hop_margin=0.10)
    tr = Tracker(10, pol)

    for _ in range(6):
        tr.update(np.full(10, 0.2))
    check("噪声不锁定", tr.target is None and tr.state == State.IDLE)

    hot = np.zeros(10); hot[2] = 0.9
    tr.update(hot)
    check("单帧不锁定", tr.target is None)
    tr.update(hot)
    check("连续两帧锁定信道3", tr.target == 2 and tr.state == State.LOCKED)

    moved = np.zeros(10); moved[7] = 0.9
    tr.update(moved)
    check("跳频第一帧仍保持旧目标", tr.target == 2)
    tr.update(moved)
    check("跳频第二帧跟随到信道8", tr.target == 7 and tr.hop_count == 1,
          f"target={tr.target} hops={tr.hop_count}")

    tr2 = Tracker(10, pol)
    a = np.zeros(10); a[0] = 0.9
    tr2.update(a); tr2.update(a)
    weak = np.zeros(10); weak[5] = 0.50
    for _ in range(4):
        tr2.update(weak)
    check("弱候选不抢目标", tr2.target == 0, f"target={tr2.target}")
    for _ in range(6):
        tr2.update(weak)
    check("目标长时间消失后释放", tr2.target is None and tr2.state == State.IDLE)

    tr3 = Tracker(10, pol)
    tr3.update(a); tr3.update(a)
    tr3.update(np.zeros(10)); tr3.update(np.zeros(10))
    blocked = np.zeros(10); blocked[5] = 0.9
    tr3.update(blocked, masked={5}); tr3.update(blocked, masked={5})
    check("遮蔽信道不被跟踪", tr3.target != 5, f"target={tr3.target}")

    tr4 = Tracker(10, pol)
    tr4.force_target(3)
    check("手动锁定信道4", tr4.target == 3)
    tr4.force_target(None)
    check("手动停止后回到空闲", tr4.target is None)


def test_sensing() -> None:
    cfg = JamConfig()
    t = np.arange(cfg.sense_samples) / cfg.sense_rate
    rng = np.random.default_rng(4)
    x = 0.02 * (rng.standard_normal(cfg.sense_samples)
                + 1j * rng.standard_normal(cfg.sense_samples))
    f = cfg.channel_centers_hz[2] - cfg.center_freq_hz
    x += 0.45 * np.exp(1j * 2 * np.pi * f * t)

    res = sense(x, cfg)
    check("占用信道评分最高", int(np.argmax(res.score)) == 2,
          f"scores={np.round(res.score, 2).tolist()}")
    check("评分超过阈值", res.score[2] >= 0.55, f"score={res.score[2]:.2f}")
    check("特征范围合法", bool(((res.features >= 0) & (res.features <= 1)).all()))

    quiet = sense(0.05 * (rng.standard_normal(cfg.sense_samples)
                          + 1j * rng.standard_normal(cfg.sense_samples)), cfg)
    check("纯噪声不误判", float(quiet.score.max()) < 0.55, f"max={quiet.score.max():.2f}")

    # 宽带占用：整段抬高、信道间没有对比度，靠绝对噪声基准识别
    rng2 = np.random.default_rng(9)
    busy = np.sqrt(1e-6 / 2) * (rng2.standard_normal(cfg.sense_samples)
                                + 1j * rng2.standard_normal(cfg.sense_samples))
    wide = sense(busy, cfg, None, noise_ref_power=1e-8)
    check("宽带占用可见", int((wide.score > 0.55).sum()) >= 6,
          f"高分信道={int((wide.score > 0.55).sum())} scores={np.round(wide.score,2).tolist()}")
    check("宽带占用判为用户", int((wide.channel_map == 1).sum()) >= 6,
          f"map={wide.channel_map.tolist()}")

    act = Action(channel_idx=2, power_db=15, bw_khz=480, waveform_mode=0)
    check("干扰覆盖信道计算", covered_channels(act, cfg) == {1, 2, 3},
          f"covered={sorted(covered_channels(act, cfg))}")

    # 中心频率搬移后，信道编号随新中心变化
    shifted = cfg.center_freq_hz + 1.6e6
    res2 = sense(x, cfg, None, center_hz=shifted)
    check("搬移中心后感知仍工作", res2.score.shape == (10,) and np.isfinite(res2.score).all())


def test_waveform() -> None:
    cfg = JamConfig()
    for mode in (0, 1):
        y, g = generate(Action(5, 12, 320, mode), cfg, 8192)
        want = min(cfg.tx_peak_max, cfg.tx_peak_min * 10 ** (12 / 20))
        check(f"波形{mode}长度", y.size == 8192)
        check(f"波形{mode}功率映射", abs(np.abs(y).max() - want) < 1e-6,
              f"peak={np.abs(y).max():.4f} 期望={want:.4f}")
        check(f"波形{mode}增益固定", abs(g - cfg.tx_gain_fixed) < 1e-9)


def test_closed_loop() -> None:
    cfg = replace(JamConfig(), jam_on_s=0.004, sense_gap_s=0.001)
    centers = cfg.channel_centers_hz
    radio = MockDualRadio(cfg, hop_plan=[(0.6, centers[6])],
                          victim_freq_hz=centers[2], seed=7)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    time.sleep(1.8)
    ctrl.stop()
    st = ctrl.status()

    check("闭环锁定初始目标", st.get("target_ch") in (3, 7), f"target={st.get('target_ch')}")
    check("闭环跟随跳频", st.get("hop_count", 0) >= 1, f"hops={st.get('hop_count')}")
    check("最终锁定新信道7", st.get("target_ch") == 7, f"target={st.get('target_ch')}")
    check("闭环持续发射", st.get("tx_frames", 0) > 20, f"tx={st.get('tx_frames')}")
    check("单周期耗时", 0 < st.get("cycle_ms", 0) < 50, f"cycle={st.get('cycle_ms')}ms")
    check("趋势历史有数据", len(ctrl.trend_history()) > 10,
          f"n={len(ctrl.trend_history())}")
    check("系统信息完整", ctrl.system_info().get("channels") == 10)


def test_recovery() -> None:
    class Flaky(MockDualRadio):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.n = 0
            self.recovered = False

        def sense_frame(self):
            self.n += 1
            if 3 <= self.n <= 7:
                raise TimeoutError("simulated")
            return super().sense_frame()

        def recover(self):
            self.recovered = True
            self.n = 0

    cfg = replace(JamConfig(), jam_on_s=0.004, sense_gap_s=0.001)
    radio = Flaky(cfg)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    time.sleep(1.5)
    ctrl.stop()
    check("接收连续失败后自恢复", radio.recovered and ctrl.status().get("frames", 0) > 0,
          f"recovered={radio.recovered} frames={ctrl.status().get('frames')}")


def test_center_scan() -> None:
    cfg = replace(JamConfig(), jam_on_s=0.004, sense_gap_s=0.001,
                  scan_interval_frames=5, scan_dwell_frames=2)
    base = cfg.center_freq_hz
    victim = cfg.channel_centers_hz[2] + 2 * 1.6e6      # 整体上移 2 个块
    radio = MockDualRadio(cfg, victim_freq_hz=victim, seed=3)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    time.sleep(2.5)
    ctrl.stop()
    st = ctrl.status()

    check("中心搬移后能搜到", st.get("target_ch") == 3, f"target={st.get('target_ch')}")
    check("中心频率已跟随", abs(st.get("center_mhz", 0) - (base + 3.2e6) / 1e6) < 0.01,
          f"center={st.get('center_mhz')} MHz")
    check("扫描后恢复发射", st.get("tx_frames", 0) > 10, f"tx={st.get('tx_frames')}")


def _wait_for(ctrl, cond, timeout: float = 3.0, poll: float = 0.03) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if cond(ctrl.status()):
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False


def test_jam_modes() -> None:
    """三种干扰模式：固定 / 随机 / 响应式。"""
    cfg = replace(JamConfig(), jam_on_s=0.004, sense_gap_s=0.001)
    centers = cfg.channel_centers_hz
    radio = MockDualRadio(cfg, hop_plan=[(0.0, centers[2])],
                          victim_freq_hz=centers[2], seed=11)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    try:
        # ── 固定干扰 ──
        ctrl.start_fixed(4)
        ok = _wait_for(ctrl, lambda s: s.get("target_ch") == 4 and s.get("jam_mode") == "fixed")
        check("固定模式已生效", ok,
              f"mode={ctrl.status().get('jam_mode')} ch={ctrl.status().get('target_ch')}")
        time.sleep(0.7)
        st = ctrl.status()
        check("固定模式目标不漂移", st.get("target_ch") == 4, f"target={st.get('target_ch')}")
        check("固定模式持续发射", st.get("tx_frames", 0) > 5, f"tx={st.get('tx_frames')}")

        # ── 随机干扰 ──
        ctrl.start_random(period_s=0.4, channels=(1, 2, 3),
                          power_min_db=5.0, power_max_db=10.0,
                          bw_choices_khz=(160.0, 320.0))
        check("随机模式已生效", _wait_for(ctrl, lambda s: s.get("jam_mode") == "random"),
              f"mode={ctrl.status().get('jam_mode')}")
        seq = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.8:     # 周期 0.6 s，留足 4 次抽样的余量
            s = ctrl.status()
            cur = (s.get("target_ch"), round(float(s.get("power_db") or 0)),
                   round(float(s.get("bw_khz") or 0)))
            if not seq or seq[-1] != cur:
                seq.append(cur)
            time.sleep(0.04)
        chans = [c[0] for c in seq if c[0]]
        check("随机模式会换信道/参数", len(seq) >= 3, f"序列={seq[:5]}")
        check("随机模式多信道轮换", len(set(chans)) >= 2, f"chans={sorted(set(chans))}")
        check("随机模式只落在许可信道", set(chans) <= {1, 2, 3}, f"chans={sorted(set(chans))}")
        check("随机功率在设定区间",
              all(5.0 <= c[1] <= 10.0 for c in seq if c[0]),
              f"power={[c[1] for c in seq][:6]}")
        check("随机带宽在可选集合",
              all(c[2] in (160, 320) for c in seq if c[0]), f"bw={[c[2] for c in seq][:6]}")
        check("随机模式持续发射", ctrl.status().get("tx_frames", 0) > 5)

        # ── 响应式干扰 ──
        ctrl.start_tracking(channel=5)
        ok = _wait_for(ctrl, lambda s: s.get("jam_mode") == "track"
                       and s.get("target_ch") == 5)
        check("响应式模式已生效", ok,
              f"mode={ctrl.status().get('jam_mode')} ch={ctrl.status().get('target_ch')}")

        ctrl.stop_tracking()
        check("停止后不再跟踪", _wait_for(ctrl, lambda s: not s.get("tracking")))
        check("停止后仍标记为响应式", ctrl.status().get("jam_mode") == "track")
    finally:
        ctrl.stop()
        radio.close()


def _centroid_hz(y: np.ndarray, fs: float) -> float:
    """频谱能量重心（Hz），用来判断基带能量到底落在哪个频率。"""
    spec = np.abs(np.fft.fft(y)) ** 2
    f = np.fft.fftfreq(len(y), 1.0 / fs)
    total = float(spec.sum())
    return float((f * spec).sum() / total) if total > 0 else 0.0


def test_tx_lo_track() -> None:
    """TX 本振跟踪：基带打 DC + 本振落在目标信道。"""
    cfg = replace(JamConfig(), jam_on_s=0.004, sense_gap_s=0.001)
    centers = cfg.channel_centers_hz

    # 基带：本振跟随时，能量重心应在 0 Hz（用多音，重心严格等于中心）
    y_dc, _ = generate(Action(0, 10, 320, 1), cfg, 4096,
                       np.random.default_rng(3), baseband_offset_hz=0.0)
    c_dc = _centroid_hz(y_dc, cfg.sense_rate)
    check("跟踪模式基带对准 DC", abs(c_dc) < 1e3, f"重心={c_dc:.0f} Hz")

    # 老做法：基带搬移到信道 1（-800 kHz）
    y_off, _ = generate(Action(0, 10, 320, 1), cfg, 4096,
                        np.random.default_rng(3), baseband_offset_hz=None)
    c_off = _centroid_hz(y_off, cfg.sense_rate)
    expect = float(centers[0] - cfg.center_freq_hz)
    check("基带搬移模式仍可用", abs(c_off - expect) < 1e3,
          f"重心={c_off:.0f} Hz 期望={expect:.0f} Hz")

    # 电台：TX 频率应等于目标信道
    radio = MockDualRadio(cfg, hop_plan=[(0.0, centers[2])],
                          victim_freq_hz=centers[2], seed=5)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    try:
        ctrl.start_fixed(1)
        ok = _wait_for(ctrl, lambda s: s.get("tx_freq_mhz") is not None)
        st = ctrl.status()
        want = float(centers[0]) / 1e6
        check("TX 本振调到信道1", ok and abs((st.get("tx_freq_mhz") or 0) - want) < 1e-3,
              f"tx={st.get('tx_freq_mhz')} MHz 期望={want:.5f} MHz")

        ctrl.start_fixed(9)
        _wait_for(ctrl, lambda s: abs((s.get("tx_freq_mhz") or 0)
                                      - float(centers[8]) / 1e6) < 1e-3, timeout=2.0)
        st = ctrl.status()
        want9 = float(centers[8]) / 1e6
        check("换信道后本振跟着搬家",
              abs((st.get("tx_freq_mhz") or 0) - want9) < 1e-3,
              f"tx={st.get('tx_freq_mhz')} MHz 期望={want9:.5f} MHz")

        # 关掉跟踪后应回落到基带搬移语义
        cfg2 = replace(cfg, tx_lo_track=False)
        radio2 = MockDualRadio(cfg2, hop_plan=[(0.0, centers[2])],
                               victim_freq_hz=centers[2], seed=5)
        ctrl2 = JamController(cfg2, radio2)
        ctrl2.start()
        try:
            ctrl2.start_fixed(1)
            time.sleep(0.5)
            st2 = ctrl2.status()
            check("关闭跟踪后不再报告 TX 频率",
                  st2.get("tx_freq_mhz") is None and st2.get("tx_lo_track") is False,
                  f"tx={st2.get('tx_freq_mhz')} track={st2.get('tx_lo_track')}")
        finally:
            ctrl2.stop()
            radio2.close()
    finally:
        ctrl.stop()
        radio.close()


def test_react_delay() -> None:
    """反应延迟：判定切频后，干扰要等设定时间才搬过去。"""
    delay = 0.4
    cfg = replace(JamConfig(), jam_on_s=0.004, sense_gap_s=0.001,
                  react_delay_s=delay)
    centers = cfg.channel_centers_hz
    radio = MockDualRadio(cfg, hop_plan=[(0.0, centers[2]), (0.8, centers[6])],
                          victim_freq_hz=centers[2], seed=9)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    t_track = t_jam = None
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < 4.0:
            st = ctrl.status()
            if t_track is None and st.get("target_ch") == 7:
                t_track = time.monotonic()
            if t_track is not None and t_jam is None and st.get("jam_target_ch") == 7:
                t_jam = time.monotonic()
                break
            time.sleep(0.02)
    finally:
        ctrl.stop()
        radio.close()

    check("延迟期内跟踪与施扰不同步",
          t_track is not None and t_jam is not None and t_jam > t_track,
          f"锁定={t_track is not None} 施扰={t_jam is not None}")
    gap = (t_jam - t_track) if (t_track and t_jam) else 0.0
    check("施扰滞后约等于设定延迟", 0.30 <= gap <= 0.75,
          f"实测 {gap * 1e3:.0f} ms，设定 {delay * 1e3:.0f} ms")


def test_long_frame() -> None:
    """长帧：一帧干扰的时长可调，且占空比要能报出来。"""
    cfg = replace(JamConfig(), jam_on_s=0.12)
    centers = cfg.channel_centers_hz
    radio = MockDualRadio(cfg, hop_plan=[(0.0, centers[5])],
                          victim_freq_hz=centers[5], seed=4)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    try:
        ctrl.start_fixed(6)
        ok = _wait_for(ctrl, lambda s: (s.get("tx_frames") or 0) >= 2, timeout=4.0)
        st = ctrl.status()
        check("帧长按设定生效", ok and abs((st.get("jam_on_ms") or 0) - 120) < 1,
              f"jam_on={st.get('jam_on_ms')} ms")
        check("占空比已上报", (st.get("duty_pct") or 0) > 50,
              f"占空比={st.get('duty_pct')}% 周期={st.get('cycle_ms')} ms")
        ctrl.set_params(jam_ms=30)
        _wait_for(ctrl, lambda s: abs((s.get("jam_on_ms") or 0) - 30) < 1, timeout=3.0)
        check("运行中可改帧长", abs((ctrl.status().get("jam_on_ms") or 0) - 30) < 1,
              f"jam_on={ctrl.status().get('jam_on_ms')} ms")
    finally:
        ctrl.stop()
        radio.close()


def test_lo_hop() -> None:
    """本振跳变：发射前搬到目标信道，发完搬回感知中心。"""

    class HopMock(MockDualRadio):
        supports_lo_hop = True

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.calls: list[float] = []

        def set_center(self, freq_hz: float) -> None:
            self.calls.append(float(freq_hz))
            super().set_center(freq_hz)

    cfg = replace(JamConfig(), jam_on_s=0.03, lo_hop=True)
    centers = cfg.channel_centers_hz
    radio = HopMock(cfg, hop_plan=[(0.0, centers[2])], victim_freq_hz=centers[2], seed=7)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    try:
        ctrl.start_fixed(3)
        ok = _wait_for(ctrl, lambda s: len(radio.calls) >= 4, timeout=4.0)
        want_ch, want_home = float(centers[2]), float(cfg.center_freq_hz)
        check("本振跳到目标信道", ok and abs(radio.calls[0] - want_ch) < 1e3,
              f"首次={radio.calls[0] / 1e6 if radio.calls else 0:.4f} MHz")
        check("发完搬回感知中心",
              len(radio.calls) > 1 and abs(radio.calls[1] - want_home) < 1e3,
              f"第二次={radio.calls[1] / 1e6 if len(radio.calls) > 1 else 0:.4f} MHz")
    finally:
        ctrl.stop()
        radio.close()


def test_park_tx() -> None:
    """待机停放：不发射时把 TX 本振挪出观测频段，一旦发射立刻解除。"""

    class ParkMock(MockDualRadio):
        def park_tx(self):
            self.tx_parked = True
            return True

    cfg = replace(JamConfig(), jam_on_s=0.02)
    centers = cfg.channel_centers_hz
    radio = ParkMock(cfg, hop_plan=[(0.0, centers[2])], victim_freq_hz=centers[2], seed=3)
    ctrl = JamController(cfg, radio)
    ctrl.start()
    try:
        ctrl.start_fixed(3)
        _wait_for(ctrl, lambda s: (s.get("tx_frames") or 0) >= 2)
        check("发射时不停放本振", not ctrl.status().get("tx_parked"),
              f"parked={ctrl.status().get('tx_parked')}")
        ctrl.stop_tracking()
        _wait_for(ctrl, lambda s: bool(s.get("tx_parked")), timeout=2.0)
        check("停止后自动停放本振", bool(ctrl.status().get("tx_parked")),
              f"parked={ctrl.status().get('tx_parked')}")
    finally:
        ctrl.stop()
        radio.close()


def main() -> int:
    print("=" * 62)
    print("干扰机自检（跟踪 / 感知 / 波形 / 闭环 / 模式 / 本振 / 时延 / 帧长 / 本振跳变 / 中心搜索）")
    print("=" * 62)
    test_tracker(); print("-" * 62)
    test_sensing(); print("-" * 62)
    test_waveform(); print("-" * 62)
    test_closed_loop(); print("-" * 62)
    test_jam_modes(); print("-" * 62)
    test_tx_lo_track(); print("-" * 62)
    test_react_delay(); print("-" * 62)
    test_long_frame(); print("-" * 62)
    test_lo_hop(); print("-" * 62)
    test_park_tx(); print("-" * 62)
    test_recovery(); print("-" * 62)
    test_center_scan()

    failed = [r for r in _results if not r[1]]
    print("=" * 62)
    print(f"合计 {len(_results)} 项，通过 {len(_results) - len(failed)}，失败 {len(failed)}")
    for name, _, detail in failed:
        print(f"  FAIL: {name} {detail}")
    return 1 if failed else 0
