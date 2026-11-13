"""干扰控制器：TDD 节拍 + 感知 + 跟踪 + 发射 + 中心频率搜索。"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import replace

import numpy as np

from .config import (JAM_MODE_NAMES, POWER_MAX_DB, POWER_MIN_DB, WAVEFORM_NAMES,
                     Action, JamConfig, JamMode, RandomPolicy)
from .dsp.sensing import SensingResult, covered_channels, sense, suppress_center_dc
from .dsp.waveform import generate
from .radio.dual import DualRadio
from .tracking.tracker import Event, Tracker, TrackerPolicy


def _clamp_power(value: float) -> float:
    """功率统一在入口处截断，避免状态里报出一个实际没用上的数值。"""
    return min(POWER_MAX_DB, max(POWER_MIN_DB, float(value)))


class JamController:
    def __init__(self, cfg: JamConfig, radio: DualRadio,
                 policy: TrackerPolicy | None = None,
                 full_duplex: bool = False,
                 on_event=None,
                 victim_plan=None):
        self.cfg = cfg
        self.radio = radio
        self.tracker = Tracker(cfg.num_channels, policy or TrackerPolicy(
            acquire_frames=cfg.acquire_frames,
            hop_confirm_frames=cfg.hop_confirm_frames,
            release_frames=cfg.release_frames,
            on_threshold=cfg.on_threshold,
            off_threshold=cfg.off_threshold,
            hop_margin=cfg.hop_margin,
        ))
        self.full_duplex = full_duplex
        self.on_event = on_event

        self._center_hz = cfg.center_freq_hz
        self._action = Action(channel_idx=0, power_db=10.0,
                              bw_khz=cfg.channel_width_hz / 1e3, waveform_mode=0)
        self._prev_action: Action | None = None
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(2026)
        self._auto_track = True
        self._hold = False
        self._jam_mode = JamMode.TRACK
        self._random = RandomPolicy()
        self._next_roll_at = 0.0
        self._last_roll: dict = {}
        self._tx_freq_hz = 0.0
        self._eff_bw_khz = float(self._action.bw_khz)
        self._eff_span = 1
        self._span_channels = max(1, min(int(cfg.span_channels), cfg.num_channels))
        self._jam_on_s = max(0.002, float(cfg.jam_on_s))
        self._duty_pct = 0.0
        self._lo_hop = bool(cfg.lo_hop)
        self._tx_parked = False
        # 反应延迟：跟踪器判定的目标与实际施扰目标之间可控的滞后
        self._react_delay_s = max(0.0, float(cfg.react_delay_s))
        self._jam_target: int | None = None
        self._react_pending: int | None = None
        self._react_detect_at = 0.0
        self.last_react_ms: float | None = None
        self._scanning = False
        self._next_scan_frame = cfg.scan_interval_frames
        self.last_hop_latency_ms: float | None = None
        self._scan_hold_until = 0.0
        self._noise_ref = float(cfg.noise_ref_power)
        self._noise_init: list[float] = []
        self._wave_preview = None
        self._wave_meta = '未发射'
        self._trend_hist: deque = deque(maxlen=900)
        self._map_hist: deque = deque(maxlen=300)
        self._rx_fail = 0
        self._started_at = time.time()
        self._victim_plan = victim_plan
        self._victim_on = False
        self._last_lost_channel: int | None = None

        self.latest: SensingResult | None = None
        self.frames = 0
        self.tx_frames = 0
        self.last_cycle_ms = 0.0
        self._spectrum: np.ndarray | None = None
        self._freqs_ghz: np.ndarray | None = None
        self._status: dict = {}

    # ── 生命周期 ──────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="jam-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── 指令接口 ─────────────────────────────────────────────
    def start_tracking(self, channel: int | None = None,
                       waveform_mode: int | None = None,
                       power_db: float | None = None,
                       bw_khz: float | None = None,
                       hold: bool = False) -> None:
        self._submit("track", dict(channel=channel, waveform_mode=waveform_mode,
                                   power_db=power_db, bw_khz=bw_khz, hold=hold))

    def start_fixed(self, channel: int, waveform_mode: int | None = None,
                    power_db: float | None = None, bw_khz: float | None = None) -> None:
        """固定信道干扰：锁死在该信道，不跟随、不自动释放。"""
        self.start_tracking(channel=channel, waveform_mode=waveform_mode,
                            power_db=power_db, bw_khz=bw_khz, hold=True)

    def start_random(self, period_s: float | None = None,
                     channels=None,
                     power_min_db: float | None = None,
                     power_max_db: float | None = None,
                     bw_choices_khz=None,
                     waveform_mode: int | None = None) -> None:
        """随机干扰：按周期随机改目标信道/功率/带宽，不参考对方通信行为。"""
        self._submit("random", dict(
            period_s=period_s, channels=channels,
            power_min_db=power_min_db, power_max_db=power_max_db,
            bw_choices_khz=bw_choices_khz, waveform_mode=waveform_mode))

    def set_center(self, freq_hz: float) -> None:
        """切换中心频率：重调 ch0 收/发，清空目标后重新搜索。"""
        self._submit("center", {"freq_hz": float(freq_hz)})

    def set_victim(self, enabled: bool) -> None:
        """开/关子板2的模拟通信信号。"""
        self._submit("victim", {"enabled": bool(enabled)})

    def stop_tracking(self) -> None:
        self._submit("stop", {})

    def set_params(self, waveform_mode: int | None = None, power_db: float | None = None,
                   bw_khz: float | None = None,
                   react_delay_ms: float | None = None,
                   span_channels: int | None = None,
                   jam_ms: float | None = None,
                   lo_hop: bool | None = None) -> None:
        self._submit("params", dict(waveform_mode=waveform_mode,
                                    power_db=power_db, bw_khz=bw_khz,
                                    react_delay_ms=react_delay_ms,
                                    span_channels=span_channels,
                                    jam_ms=jam_ms, lo_hop=lo_hop))

    def _submit(self, cmd: str, payload: dict, timeout: float = 2.0) -> bool:
        ack = threading.Event()
        self._queue.put((cmd, payload, ack))
        return ack.wait(timeout)

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    # ── 主循环 ────────────────────────────────────────────────
    def _drain_commands(self) -> None:
        while True:
            try:
                cmd, p, ack = self._queue.get_nowait()
            except queue.Empty:
                return
            if cmd == "track":
                self._auto_track = True
                self._hold = bool(p.get("hold", False))
                self._jam_mode = JamMode.FIXED if self._hold else JamMode.TRACK
                if p.get("waveform_mode") is not None:
                    self._action.waveform_mode = int(p["waveform_mode"])
                if p.get("power_db") is not None:
                    self._action.power_db = _clamp_power(p["power_db"])
                if p.get("bw_khz") is not None:
                    self._action.bw_khz = float(p["bw_khz"])
                ch = p.get("channel")
                ev = self.tracker.force_target(None if ch is None else int(ch) - 1,
                                               "manual-start")
                self._emit(ev)
                self._sync_prev_action()
            elif cmd == "stop":
                self._auto_track = False
                self._hold = False
                self._jam_mode = JamMode.TRACK
                ev = self.tracker.force_target(None, "manual-stop")
                self._emit(ev)
                self._prev_action = None
            elif cmd == "center":
                self._auto_track = True
                self._hold = False
                self._jam_mode = JamMode.TRACK
                ev = self.tracker.force_target(None, "center-change")
                self._emit(ev)
                self._prev_action = None
                self.last_hop_latency_ms = None
                try:
                    self.radio.set_center(float(p["freq_hz"]))
                    self._center_hz = float(p["freq_hz"])
                    self._tx_freq_hz = float(p["freq_hz"])   # 换中心时 TX 也被同步调走
                except Exception as exc:
                    self._publish(error=f"center: {exc}")
                self._next_scan_frame = self.frames
                # 手动换中心后先在新频段停留 10 秒，不要立刻扫走
                self._scan_hold_until = time.monotonic() + 10.0
            elif cmd == "victim":
                enabled = bool(p.get("enabled"))
                try:
                    if enabled:
                        if self._victim_plan:
                            self.radio.start_victim(self._victim_plan)
                            self._victim_on = True
                        else:
                            self._publish(error="victim: 没有模拟信号计划")
                    else:
                        self.radio.stop_victim()
                        self._victim_on = False
                except Exception as exc:
                    self._publish(error=f"victim: {exc}")
            elif cmd == "params":
                if p.get("waveform_mode") is not None:
                    self._action.waveform_mode = int(p["waveform_mode"])
                if p.get("power_db") is not None:
                    self._action.power_db = _clamp_power(p["power_db"])
                if p.get("bw_khz") is not None:
                    self._action.bw_khz = float(p["bw_khz"])
                if p.get("react_delay_ms") is not None:
                    self._react_delay_s = max(0.0, float(p["react_delay_ms"]) / 1e3)
                    self._react_pending = None    # 改了延迟，重新计时
                if p.get("span_channels") is not None:
                    self._span_channels = max(1, min(int(p["span_channels"]),
                                                     self.cfg.num_channels))
                    self._sync_prev_action()
                if p.get("jam_ms") is not None:
                    self._jam_on_s = min(max(float(p["jam_ms"]) / 1e3, 0.002), 5.0)
                if p.get("lo_hop") is not None:
                    self._lo_hop = bool(p["lo_hop"])
            elif cmd == "random":
                self._auto_track = True
                self._hold = True          # 随机模式不经过跟踪器判决
                self._jam_mode = JamMode.RANDOM
                self._apply_random_policy(p)
                if p.get("waveform_mode") is not None:
                    self._action.waveform_mode = int(p["waveform_mode"])
                self._next_roll_at = 0.0   # 立即抽第一次，避免"模式已变、参数未变"的空窗
                self._roll_random()
            self._publish()
            ack.set()

    # ── 干扰模式：随机 ────────────────────────────────────────
    def _apply_random_policy(self, p: dict) -> None:
        kw = {}
        if p.get("period_s") is not None:
            kw["period_s"] = max(0.2, float(p["period_s"]))
        if p.get("power_min_db") is not None:
            kw["power_min_db"] = float(p["power_min_db"])
        if p.get("power_max_db") is not None:
            kw["power_max_db"] = float(p["power_max_db"])
        if p.get("bw_choices_khz"):
            kw["bw_choices_khz"] = tuple(float(b) for b in p["bw_choices_khz"])
        if p.get("channels"):
            chans = sorted({min(max(int(c), 1), self.cfg.num_channels)
                            for c in p["channels"]})
            if chans:
                kw["channels"] = tuple(chans)
        pol = replace(self._random, **kw)
        if pol.power_min_db > pol.power_max_db:
            pol = replace(pol, power_min_db=pol.power_max_db,
                          power_max_db=pol.power_min_db)
        self._random = pol

    def _roll_random(self) -> None:
        """抽一组新参数并立即生效（不参考感知结果）。"""
        pol = self._random
        prev = self.tracker.target
        channel = self._pick_random_channel(prev)

        power = self._action.power_db
        if pol.vary_power:
            power = float(self._rng.uniform(pol.power_min_db, pol.power_max_db))
        bw = self._action.bw_khz
        if pol.vary_bw and pol.bw_choices_khz:
            bw = float(self._rng.choice(pol.bw_choices_khz))

        self._action.power_db = round(_clamp_power(power), 1)
        self._action.bw_khz = float(bw)
        ev = self.tracker.force_target(channel - 1, "random")
        if ev is not None:
            ev.kind = "random"          # 与"跟随"区分开：这是本地抽的，不是感知到的
            ev.detail = f"{power:.0f} dB / {bw:.0f} kHz"
        self._emit(ev)
        self._sync_prev_action()
        self._next_roll_at = time.monotonic() + max(0.2, float(pol.period_s))
        self._last_roll = {
            "channel": int(channel),
            "power_db": round(float(power), 1),
            "bw_khz": round(float(bw), 0),
            "at": round(time.time(), 3),
        }

    def _pick_random_channel(self, prev: int | None) -> int:
        pol = self._random
        chans = list(pol.channels) or list(range(1, self.cfg.num_channels + 1))
        if pol.avoid_repeat and prev is not None and len(chans) > 1:
            prev_ch = prev + 1
            filtered = [c for c in chans if c != prev_ch]
            if filtered:
                chans = filtered
        return int(self._rng.choice(chans))

    # ── 反应延迟 ──────────────────────────────────────────────
    # ── 覆盖块 ────────────────────────────────────────────────
    def _make_action(self, target: int) -> Action:
        """把目标信道展开成一个"干扰块"：N 个相邻信道 → 带宽 N×160 kHz，
        中心落在块中点（偏移 (N-1)×80 kHz），这样块边界正好对齐信道边界。"""
        cfg = self.cfg
        ch_w = cfg.channel_width_hz
        if self._jam_mode is JamMode.RANDOM:
            span = 1                      # 随机模式有自己的带宽策略
        else:
            span = min(max(int(self._span_channels), 1), cfg.num_channels - target)
        bw_khz = span * ch_w / 1e3 if span > 1 else float(self._action.bw_khz)
        offset = (span - 1) * ch_w / 2.0
        return Action(target, self._action.power_db, bw_khz,
                      self._action.waveform_mode, offset)

    def _tx_freq_for(self, act: Action) -> float:
        return float(self.cfg.centers_at(self._center_hz)[act.channel_idx]
                     + act.center_offset_hz)

    def _update_jam_target(self) -> None:
        """把"跟踪器判定"和"真正把干扰搬过去"分开，中间插入可调延迟。

        延迟期间继续压旧信道（不空着），所以对方会看到"先被压住、过一会儿才被跟上"，
        正好用来测"感知→施扰"这段时延。
        """
        tgt = self.tracker.target
        if self._jam_mode is not JamMode.TRACK or self._react_delay_s <= 0.0:
            self._jam_target = tgt
            self._react_pending = None
            return

        now = time.monotonic()
        if tgt == self._jam_target:
            self._react_pending = None
            return
        if self._react_pending != tgt:          # 又变了一次，重新计时
            self._react_pending = tgt
            self._react_detect_at = now
        waited = now - self._react_detect_at
        if waited >= self._react_delay_s:
            self._jam_target = tgt
            self.last_react_ms = round(waited * 1e3, 1)
            self._react_pending = None

    def _sync_prev_action(self) -> None:
        if self.tracker.target is None:
            self._prev_action = None
            return
        self._prev_action = self._make_action(self.tracker.target)

    def _loop(self) -> None:
        cfg = self.cfg
        while self._running:
            try:
                self._loop_once()
            except Exception as exc:
                self._publish(error=f"loop: {exc}")
                time.sleep(0.05)

    def _loop_once(self) -> None:
        cfg = self.cfg
        if True:
            t0 = time.perf_counter()
            self._drain_commands()

            # 随机干扰：到点就换一组参数，完全由本地时钟驱动
            if self._jam_mode is JamMode.RANDOM and self._auto_track:
                if time.monotonic() >= self._next_roll_at:
                    self._roll_random()

            if (self._auto_track and self.tracker.target is None
                    and not self._scanning and self.frames >= self._next_scan_frame
                    and time.monotonic() >= self._scan_hold_until):
                self._scan_once()
                self._next_scan_frame = self.frames + cfg.scan_interval_frames

            try:
                frame = self.radio.sense_frame()
            except Exception as exc:
                self._rx_fail += 1
                time.sleep(0.05)
                self._publish(error=f"rx({self._rx_fail}): {exc}")
                if self._rx_fail >= 5:
                    self._recover_radio()
                return

            self._update_spectrum(frame)
            pwr = float(np.mean(np.abs(frame) ** 2))
            if self._noise_ref <= 0:
                self._noise_init.append(pwr)
                if len(self._noise_init) >= 10:
                    self._noise_ref = float(np.median(self._noise_init))
            elif pwr < self._noise_ref * 3:
                # 只从安静帧学习，忙碌频段不会把基准抬高
                self._noise_ref = 0.98 * self._noise_ref + 0.02 * pwr
            res = sense(frame, cfg, self._prev_action, center_hz=self._center_hz,
                        noise_ref_power=self._noise_ref)
            self.latest = res
            self.frames += 1
            self._rx_fail = 0
            self._trend_hist.append((
                self.frames, round(float(res.snr_est), 1),
                round(float(res.score.max()), 3),
                (self.tracker.target + 1) if self.tracker.target is not None else 0))
            self._map_hist.append(res.channel_map.copy())

            masked = (covered_channels(self._prev_action, cfg, self._center_hz)
                      if self.full_duplex else set())
            if self.tracker.target is not None:
                masked.discard(self.tracker.target)
            events = (self.tracker.update(res.score, masked)
                      if (self._auto_track and not self._hold) else [])
            for ev in events:
                if ev.kind in ("acquired", "hop"):
                    # 仅用于测量"感知→跟随"延迟，不参与任何切换决策
                    t_hop = getattr(self.radio, "victim_last_hop_s", None)
                    if t_hop:
                        lat = (time.monotonic() - t_hop) * 1e3
                        if 0 <= lat <= 1500:      # 换频段后旧时间戳不算
                            self.last_hop_latency_ms = round(lat, 1)
                            ev.detail = (ev.detail + f" | 感知延迟 +{lat:.0f} ms").strip(" |")
                self._emit(ev)
                if ev.kind == "lost":
                    self._last_lost_channel = (ev.prev_channel - 1
                                               if ev.prev_channel else None)
                    self._next_scan_frame = self.frames + 2

            self._update_jam_target()

            target = self._jam_target
            if target is not None:
                act = self._make_action(target)
                # TX 跟踪模式：本振直接调到该信道，基带打 DC；否则基带搬移
                # lo_hop 也要求本振落在信道上，所以同样走 DC 基带
                track_lo = cfg.tx_lo_track or self._lo_hop
                tx_freq = (self._tx_freq_for(act) if track_lo else None)
                # 长帧不要重复同一小段（会在频谱上留等间隔梳状线），
                # 生成帧长的 1/8（最长 25.6 万点），其余靠平铺，梳齿间隔已低到看不见
                need = int(self._jam_on_s * cfg.sense_rate)
                gen_len = int(min(max(need // 8, 4096), 262144))
                wave, gain = generate(act, cfg, gen_len, self._rng,
                                      center_hz=self._center_hz,
                                      baseband_offset_hz=0.0 if tx_freq is not None else None)
                self._eff_bw_khz = act.bw_khz
                self._eff_span = int(round(act.bw_khz * 1e3 / cfg.channel_width_hz))
                hop = bool(self._lo_hop and self.radio.supports_lo_hop)
                try:
                    if hop:
                        # 共用本振：整块搬到目标信道，基带就在 DC
                        self.radio.set_center(float(tx_freq))
                    self.radio.jam(wave, gain, self._jam_on_s, tx_freq_hz=tx_freq)
                    self.tx_frames += 1
                    self._tx_parked = False
                    if tx_freq is not None:
                        # 记实测值：UHD 调谐可能留几 kHz 残差，报告要能对上频谱仪
                        actual = self.radio.tx_freq()
                        self._tx_freq_hz = float(actual) if actual else tx_freq
                except Exception as exc:
                    self._publish(error=f"tx: {exc}")
                finally:
                    if hop:
                        # 发完立刻搬回感知中心：否则 3.000000 上会常驻一根本振泄漏
                        try:
                            self.radio.set_center(self._center_hz)
                            self.radio.flush_rx()
                        except Exception as exc:
                            self._publish(error=f"lo-hop: {exc}")
                self._prev_action = act
                self._update_wave_preview(wave, act)
            else:
                self._prev_action = None
                self._wave_preview = None
                self._wave_meta = "未发射"
                # 待机时把 TX 本振挪出观测频段，否则泄漏谱线正好压在 3 GHz 上
                if not self._tx_parked:
                    try:
                        self._tx_parked = bool(self.radio.park_tx())
                    except Exception:
                        self._tx_parked = False
                time.sleep(min(self._jam_on_s, 0.02))   # 没目标时快速轮询感知

            time.sleep(cfg.sense_gap_s)
            try:
                self.radio.flush_rx()
            except Exception:
                pass

            self.last_cycle_ms = (time.perf_counter() - t0) * 1e3
            if self.last_cycle_ms > 0 and self.tx_frames:
                self._duty_pct = min(100.0,
                                     self._jam_on_s * 1e3 / self.last_cycle_ms * 100.0)
            self._publish()

    # ── 历史与自恢复 ──────────────────────────────────────────
    def trend_history(self, frames: int = 120) -> list[tuple]:
        return list(self._trend_hist)[-max(0, int(frames)):]

    def map_history(self, frames: int = 40) -> list[list[int]]:
        return [m.tolist() for m in list(self._map_hist)[-max(0, int(frames)):]]

    def system_info(self) -> dict:
        st = self.status()
        return {
            "center_mhz": st.get("center_mhz", 0),
            "channels": self.cfg.num_channels,
            "channel_khz": self.cfg.channel_width_hz / 1e3,
            "sample_rate_msps": self.cfg.sense_rate / 1e6,
            "rx_port": f"ch{self.cfg.rx_ch} {self.cfg.rx_antenna}",
            "tx_port": f"ch{self.cfg.tx_ch} {self.cfg.tx_antenna}",
            "victim_port": f"ch{self.cfg.victim_ch} {self.cfg.victim_antenna}",
            "tx_gain_db": self.cfg.tx_gain_fixed,
            "jam_ms": self.cfg.jam_on_s * 1e3,
            "cycle_ms": st.get("cycle_ms", 0),
            "thresholds": [self.cfg.on_threshold, self.cfg.off_threshold],
            "release_frames": self.cfg.release_frames,
            "uptime_s": round(time.time() - self._started_at, 1),
            "tx_frames": st.get("tx_frames", 0),
            "mode": st.get("mode", "-"),
            "jam_mode": st.get("jam_mode", "-"),
            "jam_mode_name": st.get("jam_mode_name", "-"),
            "random_policy": st.get("random_policy", "-"),
            "tx_lo_track": st.get("tx_lo_track"),
            "tx_freq_mhz": st.get("tx_freq_mhz"),
            "react_delay_ms": st.get("react_delay_ms"),
        }

    def _recover_radio(self) -> None:
        try:
            self.radio.recover()
            self._rx_fail = 0
            self._publish(error="设备已尝试重新打开")
        except Exception as exc:
            self._publish(error=f"recover: {exc}")

    def _update_wave_preview(self, wave: np.ndarray, act: Action) -> None:
        step = max(1, len(wave) // 512)
        self._wave_preview = np.abs(wave[::step])[:512]
        self._wave_meta = (f"{WAVEFORM_NAMES.get(act.waveform_mode, '-')} · "
                           f"{act.bw_khz:.0f} kHz · {act.power_db:.0f} dB")

    # ── 中心频率搜索 ──────────────────────────────────────────
    def _scan_once(self) -> None:
        cfg = self.cfg
        base = self._center_hz
        self._scanning = True
        self._publish()
        found = None
        off = 0.0
        try:
            for off in cfg.scan_offsets_hz:
                center = base + off
                self.radio.set_center(center)
                self._center_hz = center
                self.radio.flush_rx()          # 清掉上一轮发射残留
                hits = np.zeros(cfg.num_channels, dtype=int)
                for _ in range(cfg.scan_dwell_frames):
                    try:
                        frame = self.radio.sense_frame()
                    except Exception:
                        continue
                    res = sense(frame, cfg, None, center_hz=center,
                                noise_ref_power=self._noise_ref)
                    self.latest = res
                    self._update_spectrum(frame)
                    hits += (res.score >= cfg.on_threshold).astype(int)
                # 刚被释放的信道在原点位不立即重锁，避免把自己的发射尾迹当成目标
                if off == 0 and self._last_lost_channel is not None:
                    hits[self._last_lost_channel] = 0
                if hits.max() >= 2:
                    found = int(np.argmax(hits))
                    break
        finally:
            if found is None:
                self.radio.set_center(base)
                self._center_hz = base
            self._scanning = False

        if found is not None:
            ev = self.tracker.force_target(found, f"scan {off / 1e6:+.1f} MHz")
            self._emit(ev)
            self._sync_prev_action()
        self._publish()

    # ── 显示用频谱 ────────────────────────────────────────────
    def _update_spectrum(self, frame: np.ndarray) -> None:
        n = frame.size
        x = np.asarray(frame, dtype=np.complex128).ravel()
        x = x - x.mean()
        mag = np.abs(np.fft.fftshift(np.fft.fft(x)))
        mag = suppress_center_dc(mag, self.cfg)
        step = max(1, n // 512)
        self._spectrum = mag[::step]
        freqs = self._center_hz + (np.arange(n) - n / 2) * (self.cfg.sense_rate / n)
        self._freqs_ghz = freqs[::step] / 1e9

    def _jam_coverage(self) -> list[float]:
        """当前干扰动作对 10 个信道的覆盖比例 0~1，用于界面叠加显示。"""
        a = self._prev_action
        cfg = self.cfg
        if a is None or a.power_db <= 0:
            return [0.0] * cfg.num_channels
        centers = cfg.centers_at(self._center_hz)
        half = cfg.channel_width_hz / 2
        bw = a.bw_khz * 1e3
        lo = centers[a.channel_idx] - bw / 2
        hi = centers[a.channel_idx] + bw / 2
        out = []
        for k in range(cfg.num_channels):
            ov = max(0.0, min(hi, centers[k] + half) - max(lo, centers[k] - half))
            out.append(round(ov / cfg.channel_width_hz, 3))
        return out

    # ── 状态发布 ──────────────────────────────────────────────
    def _emit(self, ev: Event | None) -> None:
        if ev is not None and self.on_event:
            self.on_event(ev)

    def _publish(self, error: str | None = None) -> None:
        t = self.tracker
        res = self.latest
        st = {
            "running": self._running,
            "tracking": self._auto_track,
            "hold": self._hold,
            "scanning": self._scanning,
            "state": t.state.value,
            "jam_mode": self._jam_mode.value,
            "jam_mode_name": JAM_MODE_NAMES.get(self._jam_mode.value, "-"),
            "mode": ("stopped" if not self._auto_track
                     else self._jam_mode.value if self._jam_mode is not JamMode.TRACK
                     else "auto"),
            "random_period_s": round(float(self._random.period_s), 2),
            "random_next_s": (round(max(0.0, self._next_roll_at - time.monotonic()), 1)
                              if self._jam_mode is JamMode.RANDOM and self._auto_track
                              else None),
            "random_last": dict(self._last_roll),
            "random_policy": self._random.described(),
            "react_delay_ms": round(self._react_delay_s * 1e3, 1),
            "react_active": (self._jam_mode is JamMode.TRACK
                             and self._react_delay_s > 0),
            "react_pending_ms": (
                round(max(0.0, self._react_delay_s
                          - (time.monotonic() - self._react_detect_at)) * 1e3, 1)
                if (self._react_pending is not None and self._react_delay_s > 0)
                else None),
            "last_react_ms": self.last_react_ms,
            "jam_target_ch": ((self._jam_target + 1)
                              if self._jam_target is not None else None),
            "victim_on": self._victim_on,
            "hop_latency_ms": self.last_hop_latency_ms,
            "noise_ref": round(self._noise_ref, 12) if self._noise_ref else 0.0,
            "wave_meta": self._wave_meta,
            "wave_preview": (self._wave_preview.tolist()
                             if self._wave_preview is not None else []),
            "target_ch": (t.target + 1) if t.target is not None else None,
            "hop_count": t.hop_count,
            "frames": self.frames,
            "tx_frames": self.tx_frames,
            "cycle_ms": round(self.last_cycle_ms, 2),
            "jam_on_ms": round(self._jam_on_s * 1e3, 1),
            "duty_pct": round(self._duty_pct, 1),
            "center_mhz": round(self._center_hz / 1e6, 4),
            "tx_freq_mhz": round(self._tx_freq_hz / 1e6, 5) if self._tx_freq_hz else None,
            "tx_lo_track": bool(self.cfg.tx_lo_track),
            "lo_hop": bool(self._lo_hop),
            "lo_independent": getattr(self.radio, "lo_independent", None),
            "tx_parked": bool(self._tx_parked),
            "tx_tune_ms": round(float(getattr(self.radio, "last_tune_ms", 0.0)), 2),
            "tx_err_hz": (round(self._tx_freq_hz - self._tx_freq_for(
                self._make_action(t.target)), 1)
                if (self.cfg.tx_lo_track and t.target is not None
                    and self._tx_freq_hz) else None),
            "waveform": self._action.waveform_mode,
            "power_db": self._action.power_db,
            "bw_khz": round(self._eff_bw_khz, 1),
            "bw_set_khz": self._action.bw_khz,
            "span_channels": self._eff_span,
            "span_to_ch": (min(10, (self._jam_target or 0) + self._eff_span)
                           if self._jam_target is not None else None),
            "scores": [round(float(v), 3) for v in (res.score if res else [])],
            "jam_cov": self._jam_coverage(),
            "channel_map": [int(v) for v in (res.channel_map if res else [])],
            "snr_est": round(float(res.snr_est), 2) if res else 0.0,
            "last_event": vars(t.events[-1]) if t.events else None,
        }
        if error:
            st["error"] = error
        with self._lock:
            st["spectrum"] = (self._spectrum.tolist()
                              if self._spectrum is not None else [])
            st["freqs_ghz"] = (self._freqs_ghz.tolist()
                               if self._freqs_ghz is not None else [])
            self._status = st
