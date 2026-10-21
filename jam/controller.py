"""干扰控制器：TDD 节拍 + 感知 + 跟踪 + 发射 + 中心频率搜索。"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .config import WAVEFORM_NAMES, Action, JamConfig
from .dsp.sensing import SensingResult, covered_channels, sense, suppress_center_dc
from .dsp.waveform import generate
from .radio.dual import DualRadio
from .tracking.tracker import Event, Tracker, TrackerPolicy


class JamController:
    def __init__(self, cfg: JamConfig, radio: DualRadio,
                 policy: TrackerPolicy | None = None,
                 full_duplex: bool = False,
                 on_event=None):
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
        self._scanning = False
        self._next_scan_frame = cfg.scan_interval_frames
        self.last_hop_latency_ms: float | None = None
        self._scan_hold_until = 0.0
        self._noise_ref = float(cfg.noise_ref_power)
        self._noise_init: list[float] = []
        self._wave_preview = None
        self._wave_meta = '未发射'
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

    def set_center(self, freq_hz: float) -> None:
        """切换中心频率：重调 ch0 收/发，清空目标后重新搜索。"""
        self._submit("center", {"freq_hz": float(freq_hz)})

    def stop_tracking(self) -> None:
        self._submit("stop", {})

    def set_params(self, waveform_mode: int | None = None, power_db: float | None = None,
                   bw_khz: float | None = None) -> None:
        self._submit("params", dict(waveform_mode=waveform_mode,
                                    power_db=power_db, bw_khz=bw_khz))

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
                if p.get("waveform_mode") is not None:
                    self._action.waveform_mode = int(p["waveform_mode"])
                if p.get("power_db") is not None:
                    self._action.power_db = float(p["power_db"])
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
                ev = self.tracker.force_target(None, "manual-stop")
                self._emit(ev)
                self._prev_action = None
            elif cmd == "center":
                self._auto_track = True
                self._hold = False
                ev = self.tracker.force_target(None, "center-change")
                self._emit(ev)
                self._prev_action = None
                self.last_hop_latency_ms = None
                try:
                    self.radio.set_center(float(p["freq_hz"]))
                    self._center_hz = float(p["freq_hz"])
                except Exception as exc:
                    self._publish(error=f"center: {exc}")
                self._next_scan_frame = self.frames
                # 手动换中心后先在新频段停留 10 秒，不要立刻扫走
                self._scan_hold_until = time.monotonic() + 10.0
            elif cmd == "params":
                if p.get("waveform_mode") is not None:
                    self._action.waveform_mode = int(p["waveform_mode"])
                if p.get("power_db") is not None:
                    self._action.power_db = float(p["power_db"])
                if p.get("bw_khz") is not None:
                    self._action.bw_khz = float(p["bw_khz"])
            self._publish()
            ack.set()

    def _sync_prev_action(self) -> None:
        if self.tracker.target is None:
            self._prev_action = None
            return
        self._prev_action = Action(self.tracker.target, self._action.power_db,
                                   self._action.bw_khz, self._action.waveform_mode)

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

            if (self._auto_track and self.tracker.target is None
                    and not self._scanning and self.frames >= self._next_scan_frame
                    and time.monotonic() >= self._scan_hold_until):
                self._scan_once()
                self._next_scan_frame = self.frames + cfg.scan_interval_frames

            try:
                frame = self.radio.sense_frame()
            except Exception as exc:
                time.sleep(0.05)
                self._publish(error=f"rx: {exc}")
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

            target = self.tracker.target
            if target is not None:
                act = Action(target, self._action.power_db, self._action.bw_khz,
                             self._action.waveform_mode)
                wave, gain = generate(act, cfg, cfg.jam_block_samples, self._rng,
                                      center_hz=self._center_hz)
                try:
                    self.radio.jam(wave, gain, cfg.jam_on_s)
                    self.tx_frames += 1
                except Exception as exc:
                    self._publish(error=f"tx: {exc}")
                self._prev_action = act
                self._update_wave_preview(wave, act)
            else:
                self._prev_action = None
                self._wave_preview = None
                self._wave_meta = "未发射"
                time.sleep(cfg.jam_on_s)

            time.sleep(cfg.sense_gap_s)
            try:
                self.radio.flush_rx()
            except Exception:
                pass

            self.last_cycle_ms = (time.perf_counter() - t0) * 1e3
            self._publish()

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
            "mode": ("fixed" if self._hold else
                     ("auto" if self._auto_track else "stopped")),
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
            "center_mhz": round(self._center_hz / 1e6, 4),
            "waveform": self._action.waveform_mode,
            "power_db": self._action.power_db,
            "bw_khz": self._action.bw_khz,
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
