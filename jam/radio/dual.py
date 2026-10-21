"""双子板射频层。

端口分配（X310 + 两块 UBX，四根天线）：
  子板 A / ch0  RX2    → 专职感知接收
  子板 B / ch1  TX/RX  → 专职干扰发射
  中心频率可在运行中重调，用于跟随通信方整体搬移的 10 信道栅格。

现场约束：天线间距约 5 cm、无衰减器。实测 0 dB 增益 + 满幅基带时，
对端接收幅度仅 ~0.006，无削顶；因此 TX 增益固定 0，功率完全由基带幅度控制，
并且严格采用"先收后发"的 TDD 节拍，发射期间不采样。
"""
from __future__ import annotations

import gc
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from ..config import JamConfig


class DualRadio(ABC):
    center_hz: float

    @abstractmethod
    def sense_frame(self) -> np.ndarray:
        ...

    @abstractmethod
    def jam(self, samples: np.ndarray, gain_db: float, duration_s: float) -> None:
        ...

    def set_center(self, freq_hz: float) -> None:
        self.center_hz = freq_hz

    def flush_rx(self) -> None:
        ...

    def recover(self) -> None:
        ...

    def close(self) -> None:
        ...


class MockDualRadio(DualRadio):
    """无硬件仿真。

    victim_freq_hz 是通信方的绝对频点；中心频率搬移后，只有它落在
    当前 1.6 MHz 窗口内才会被"看见"，用于验证中心频率搜索。
    hop_plan: [(秒, 绝对频率 Hz), ...]
    """

    def __init__(self, cfg: JamConfig, hop_plan: list[tuple[float, float]] | None = None,
                 seed: int = 1, victim_freq_hz: float | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.t = np.arange(cfg.sense_samples) / cfg.sense_rate
        self.center_hz = cfg.center_freq_hz
        self.victim_freq_hz = (victim_freq_hz
                               if victim_freq_hz is not None
                               else cfg.channel_centers_hz[2])   # 默认信道3
        self.hop_plan = sorted(hop_plan or [], key=lambda x: x[0])
        self._t0 = time.monotonic()
        self._hop_idx = 0
        self.victim_last_hop_s = time.monotonic()
        self.jam_cycles = 0
        self.last_jam_channel: int | None = None
        self.tx_enabled = True
        self.center_history: list[float] = []

    def _apply_schedule(self) -> None:
        elapsed = time.monotonic() - self._t0
        while self._hop_idx < len(self.hop_plan) and elapsed >= self.hop_plan[self._hop_idx][0]:
            self.victim_freq_hz = self.hop_plan[self._hop_idx][1]
            self._hop_idx += 1
            self.victim_last_hop_s = time.monotonic()

    def set_center(self, freq_hz: float) -> None:
        self.center_hz = freq_hz
        self.center_history.append(freq_hz)

    def sense_frame(self) -> np.ndarray:
        self._apply_schedule()
        cfg = self.cfg
        x = 0.02 * (self.rng.standard_normal(cfg.sense_samples)
                    + 1j * self.rng.standard_normal(cfg.sense_samples))

        off = self.victim_freq_hz - self.center_hz
        if abs(off) <= cfg.sense_rate / 2 - 2 * cfg.channel_width_hz:
            x += 0.45 * np.exp(1j * 2 * np.pi * off * self.t)
            second = off + 4 * cfg.channel_width_hz
            if abs(second) <= cfg.sense_rate / 2 - cfg.channel_width_hz:
                x += 0.18 * np.exp(1j * 2 * np.pi * second * self.t)
        return x

    def jam(self, samples: np.ndarray, gain_db: float, duration_s: float) -> None:
        self.jam_cycles += 1
        time.sleep(duration_s)


class UhdDualRadio(DualRadio):
    def __init__(self, cfg: JamConfig, args: str | None = None,
                 tx_enabled: bool = True, rx_thread: bool = False):
        import uhd

        self.uhd = uhd
        self.cfg = cfg
        self.tx_enabled = tx_enabled
        self.rx_thread = rx_thread
        self.overflow = 0
        self._tx_end_time = 0.0
        self.center_hz = cfg.center_freq_hz
        self._reader = None
        self._vtx = None
        self.victim_last_hop_s = None
        self._victim_stop = None
        self._victim_thread = None
        self._victim_plan: list[tuple[float, float]] = []
        self._victim_amp = 0.05
        self._args = args or f"addr={cfg.ip_addr}"

        self.usrp = uhd.usrp.MultiUSRP(self._args)
        self._configure()
        self._open_streams()

    def _configure(self) -> None:
        uhd, cfg = self.uhd, self.cfg
        self.usrp.set_rx_rate(cfg.sense_rate, cfg.rx_ch)
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(cfg.center_freq_hz), cfg.rx_ch)
        self.usrp.set_rx_gain(cfg.rx_gain, cfg.rx_ch)
        self.usrp.set_rx_antenna(cfg.rx_antenna, cfg.rx_ch)

        self.usrp.set_tx_rate(cfg.sense_rate, cfg.tx_ch)
        self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(cfg.center_freq_hz), cfg.tx_ch)
        self.usrp.set_tx_gain(cfg.tx_gain_fixed, cfg.tx_ch)
        self.usrp.set_tx_antenna(cfg.tx_antenna, cfg.tx_ch)

        for mb in range(self.usrp.get_num_mboards()):
            self.usrp.set_clock_source("internal", mb)

    def _open_streams(self) -> None:
        uhd, cfg = self.uhd, self.cfg
        sa = uhd.usrp.StreamArgs("fc32", "sc16")
        sa.channels = [cfg.rx_ch]
        self._rx = self.usrp.get_rx_stream(sa)

        sa_tx = uhd.usrp.StreamArgs("fc32", "sc16")
        sa_tx.channels = [cfg.tx_ch]
        self._tx = self.usrp.get_tx_stream(sa_tx)

        self._buf = np.zeros((1, cfg.sense_samples * 2), dtype=np.complex64)
        self._rx_md = uhd.types.RXMetadata()
        self._tx_md = uhd.types.TXMetadata()

        self._latest: tuple[float, bool, np.ndarray] | None = None
        self._latest_lock = threading.Lock()
        self._min_time = 0.0
        self._reader_stop = threading.Event()

        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self._rx.issue_stream_cmd(cmd)
        if self.rx_thread:
            # 起排空线程避免缓冲溢出；用时间门槛跳过启动 DC 瞬态
            self._min_time = self.usrp.get_time_now().get_real_secs() + 0.05
            self._reader = threading.Thread(target=self._rx_loop, name="uhd-rx", daemon=True)
            self._reader.start()
        else:
            # 环回模式：ch0 同时收发，线程化接收会和发射流冲突，改回同步收
            self.drain(cfg.sense_warmup_frames)

    def _rx_loop(self) -> None:
        """持续排空 RX 流，只保留最新一帧。

        2 MS/s 的数据如果每轮只取 4096 点，剩余样本会在 UHD 缓冲里堆积并持续
        溢出（控制台那串 O）。这里用独立线程全速消费，感知只读最新帧。
        """
        while not self._reader_stop.is_set():
            try:
                got = self._rx.recv(self._buf, self._rx_md, 0.1)
                if self._rx_md.error_code == self.uhd.types.RXMetadataErrorCode.overflow:
                    self.overflow += 1
                    continue
                if not got:
                    continue
                has_ts = self._rx_md.has_time_spec
                ts = self._rx_md.time_spec.get_real_secs() if has_ts else 0.0
                with self._latest_lock:
                    self._latest = (ts, has_ts, self._buf[0, :got].copy())
            except Exception:
                time.sleep(0.01)

    def drain(self, frames: int = 3) -> None:
        for _ in range(frames):
            self._rx.recv(self._buf, self._rx_md, 0.1)

    # ── 收 ────────────────────────────────────────────────────
    def sense_frame(self) -> np.ndarray:
        cfg = self.cfg
        if not self.rx_thread:
            # 限时排空积压，避免 recv(0.0) 在无数据时反复阻塞把周期拖长
            deadline = time.monotonic() + 0.01
            while time.monotonic() < deadline:
                try:
                    if not self._rx.recv(self._buf, self._rx_md, 0.0):
                        break
                except RuntimeError:
                    break
            out = np.empty(cfg.sense_samples, dtype=np.complex64)
            filled, deadline = 0, time.monotonic() + 1.0
            while filled < cfg.sense_samples:
                if time.monotonic() > deadline:
                    raise TimeoutError("RX timeout")
                got = self._rx.recv(self._buf, self._rx_md, 0.1)
                if self._rx_md.error_code == self.uhd.types.RXMetadataErrorCode.overflow:
                    self.overflow += 1
                    continue
                if got:
                    take = min(got, cfg.sense_samples - filled)
                    out[filled:filled + take] = self._buf[0, :take]
                    filled += take
            return out.astype(np.complex128)

        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            with self._latest_lock:
                item = self._latest
            if item is not None:
                ts, has_ts, samples = item
                if samples.size >= cfg.sense_samples and (not has_ts or ts >= self._min_time):
                    return samples[:cfg.sense_samples].astype(np.complex128)
            time.sleep(0.001)
        raise TimeoutError("RX timeout")

    def flush_rx(self) -> None:
        """丢弃发射窗口内的样本：线程模式设时间门槛，同步模式直接排空。"""
        if not self._tx_end_time:
            return
        if self.rx_thread:
            self._min_time = max(self._min_time, self._tx_end_time + 0.002)
            return
        try:
            deadline = time.monotonic() + 0.05
            while time.monotonic() < deadline:
                got = self._rx.recv(self._buf, self._rx_md, 0.005)
                if not got:
                    break
                if self._rx_md.has_time_spec and \
                        self._rx_md.time_spec.get_real_secs() > self._tx_end_time + 0.002:
                    break
        except RuntimeError:
            return

    # ── 发 ────────────────────────────────────────────────────
    def jam(self, samples: np.ndarray, gain_db: float, duration_s: float) -> None:
        if not self.tx_enabled:
            return
        cfg = self.cfg
        need = int(round(duration_s * cfg.sense_rate))
        block = np.asarray(samples, dtype=np.complex64).ravel()
        if block.size == 0:
            return
        reps = int(np.ceil(need / block.size))
        burst = np.tile(block, reps)[:need].reshape(1, -1)

        # 分块发送，避免单次大缓冲把 UHD 控制通道压到 ACK 超时
        chunk = 8192
        total = burst.shape[1]
        self._tx_md.has_time_spec = False
        for start in range(0, total, chunk):
            part = np.ascontiguousarray(burst[:, start:start + chunk])
            self._tx_md.start_of_burst = (start == 0)
            self._tx_md.end_of_burst = (start + chunk >= total)
            self._tx.send(part, self._tx_md, 1.0)
        # send() 返回时突发已基本播完，用此刻设备时间作为静默起点更可靠
        self._tx_end_time = (self.usrp.get_time_now().get_real_secs()
                             + duration_s + 0.005)

    # ── 内置模拟通信信号（ch0 TX/RX 口）──────────────────────
    def start_victim(self, plan: list[tuple[float, float]], amp: float = 0.05) -> None:
        """用 ch0 的 TX/RX 口发一个按计划跳频的单音，充当"通信方"。

        plan: [(秒, 绝对频率 Hz), ...]。仅用于环回演示/自测。
        """
        uhd, cfg = self.uhd, self.cfg
        if not plan:
            return
        self._victim_plan = sorted(plan, key=lambda x: x[0])
        self._victim_amp = float(amp)
        self.usrp.set_tx_rate(cfg.sense_rate, cfg.victim_ch)
        self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(cfg.center_freq_hz),
                              cfg.victim_ch)
        self.usrp.set_tx_gain(0.0, cfg.victim_ch)
        self.usrp.set_tx_antenna(cfg.victim_antenna, cfg.victim_ch)

        sa = uhd.usrp.StreamArgs("fc32", "sc16")
        sa.channels = [cfg.victim_ch]
        self._vtx = self.usrp.get_tx_stream(sa)
        # 预先灌 4 个块（约 64 ms），避免启动瞬间欠载刷 U
        tt = np.arange(32768) / cfg.sense_rate
        off0 = self._victim_plan[0][1] - cfg.center_freq_hz
        blk0 = (amp * np.exp(1j * 2 * np.pi * off0 * tt)).astype(np.complex64)
        md0 = uhd.types.TXMetadata()
        md0.start_of_burst = False
        md0.end_of_burst = False
        for _ in range(4):
            try:
                self._vtx.send(blk0.reshape(1, -1), md0, 0.5)
            except Exception:
                break
        self.victim_last_hop_s = time.monotonic()
        self._victim_stop = threading.Event()
        self._victim_thread = threading.Thread(target=self._victim_loop,
                                               args=(amp,), name="victim-tx", daemon=True)
        self._victim_thread.start()

    def _victim_loop(self, amp: float) -> None:
        t = np.arange(32768) / self.cfg.sense_rate
        md = self.uhd.types.TXMetadata()
        md.start_of_burst = False
        md.end_of_burst = False
        t0 = time.monotonic()
        idx = 0
        while not self._victim_stop.is_set():
            elapsed = time.monotonic() - t0
            while idx + 1 < len(self._victim_plan) and elapsed >= self._victim_plan[idx + 1][0]:
                idx += 1
                self.victim_last_hop_s = time.monotonic()
            # 走完一轮后停 6 秒再从头开始，方便长时间演示
            if idx == len(self._victim_plan) - 1 and elapsed >= self._victim_plan[-1][0] + 6.0:
                t0 = time.monotonic()
                idx = 0
                elapsed = 0.0
                self.victim_last_hop_s = time.monotonic()
            off = self._victim_plan[idx][1] - self.cfg.center_freq_hz
            blk = (amp * np.exp(1j * 2 * np.pi * off * t)).astype(np.complex64)
            try:
                self._vtx.send(blk.reshape(1, -1), md, 0.5)
            except Exception:
                time.sleep(0.01)

    # ── 调谐 ──────────────────────────────────────────────────
    def set_center(self, freq_hz: float) -> None:
        uhd = self.uhd
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq_hz), self.cfg.rx_ch)
        self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(freq_hz), self.cfg.tx_ch)
        self.center_hz = freq_hz
        if self.rx_thread:
            self._min_time = max(self._min_time,
                                 self.usrp.get_time_now().get_real_secs() + 0.02)
        else:
            # 本振重调后会有几十毫秒的瞬态，多丢一些帧，避免被当成信号锁定
            time.sleep(0.02)
            self.drain(20)

    def recover(self) -> None:
        """RX 连续失败时尝试重开数据流，保留当前中心频率和模拟信号计划。"""
        plan = list(self._victim_plan)
        amp = self._victim_amp
        center = self.center_hz

        self._reader_stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        if self._victim_stop is not None:
            self._victim_stop.set()
            if self._victim_thread is not None:
                self._victim_thread.join(timeout=1.0)
                self._victim_thread = None
            self._vtx = None
        try:
            cmd = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.stop_cont)
            cmd.stream_now = True
            self._rx.issue_stream_cmd(cmd)
            time.sleep(0.15)
        except Exception:
            pass

        self._rx = self._tx = None
        self._tx_end_time = 0.0
        self._min_time = 0.0
        self._latest = None
        time.sleep(0.2)

        self._configure()
        self._open_streams()
        self.set_center(center)
        if plan:
            self.start_victim(plan, amp)

    def close(self) -> None:
        if self._victim_stop is not None:
            self._victim_stop.set()
            if self._victim_thread is not None:
                self._victim_thread.join(timeout=1.0)
                self._victim_thread = None
            self._vtx = None
        self._reader_stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        try:
            cmd = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.stop_cont)
            cmd.stream_now = True
            self._rx.issue_stream_cmd(cmd)
            time.sleep(0.2)      # 给设备处理 stop 的时间
        except Exception:
            pass
        self._rx = self._tx = None
        self.usrp = None
        gc.collect()
        time.sleep(0.2)


def make_radio(kind: str, cfg: JamConfig, hop_plan=None,
               tx_enabled: bool = True, rx_thread: bool = False) -> DualRadio:
    if kind == "uhd":
        return UhdDualRadio(cfg, tx_enabled=tx_enabled, rx_thread=rx_thread)
    return MockDualRadio(cfg, hop_plan=hop_plan)
