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
        check(f"波形{mode}长度", y.size == 8192)
        check(f"波形{mode}功率映射", abs(np.abs(y).max() - 0.25 * 10 ** (-8 / 20)) < 1e-9,
              f"peak={np.abs(y).max():.4f}")
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


def main() -> int:
    print("=" * 62)
    print("干扰机自检（跟踪 / 感知 / 波形 / 闭环 / 中心搜索）")
    print("=" * 62)
    test_tracker(); print("-" * 62)
    test_sensing(); print("-" * 62)
    test_waveform(); print("-" * 62)
    test_closed_loop(); print("-" * 62)
    test_recovery(); print("-" * 62)
    test_center_scan()

    failed = [r for r in _results if not r[1]]
    print("=" * 62)
    print(f"合计 {len(_results)} 项，通过 {len(_results) - len(failed)}，失败 {len(failed)}")
    for name, _, detail in failed:
        print(f"  FAIL: {name} {detail}")
    return 1 if failed else 0
