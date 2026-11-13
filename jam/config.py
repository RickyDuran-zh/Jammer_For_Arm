"""全局配置：1.6 MHz 观测带宽、10×160 kHz 信道、双子板端口映射。

功率策略（现场 5 cm 天线间距、无衰减器，实测得出）：
  TX 增益固定 0 dB，干扰强度全部由基带幅度表达。
  实测满幅 0.25 时对端接收仅 ~0.006（无削顶），留足安全余量；
  因此不做 RF 增益调节，避免误操作把 UBX 前端压死。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

BW_CHOICES_KHZ: tuple[float, ...] = (160.0, 320.0, 480.0, 640.0, 1600.0)
POWER_MIN_DB: float = -30.0
POWER_MAX_DB: float = 20.0
WAVEFORM_NAMES: dict[int, str] = {0: "限带噪声", 1: "多音"}

CH_IDLE, CH_USER, CH_JAMMED, CH_OVERLAP = 0, 1, 2, 3


class JamMode(str, Enum):
    """三种干扰模式（对应申报书表 2 的对抗场景）。"""
    FIXED = "fixed"      # 固定干扰：锁死一个信道，参数不变
    TRACK = "track"      # 响应式干扰：感知→锁定→跟随对方切频
    RANDOM = "random"    # 随机干扰：按周期随机改信道/功率/带宽，不看对方动作


JAM_MODE_NAMES: dict[str, str] = {
    JamMode.FIXED.value: "固定干扰",
    JamMode.TRACK.value: "响应式干扰",
    JamMode.RANDOM.value: "随机干扰",
}


@dataclass(frozen=True)
class RandomPolicy:
    """随机干扰的参数空间。

    申报书表 2：随机时变干扰"按设定周期随机改变目标频点、干扰功率或干扰带宽，
    各次参数变化不以蓝方通信动作为依据"。
    """
    period_s: float = 2.0
    channels: tuple[int, ...] = tuple(range(1, 11))   # 1-based
    power_min_db: float = 5.0
    power_max_db: float = 20.0
    bw_choices_khz: tuple[float, ...] = (160.0, 320.0, 480.0)
    vary_channel: bool = True
    vary_power: bool = True
    vary_bw: bool = True
    avoid_repeat: bool = True     # 新信道尽量不等于上一个

    def described(self) -> str:
        parts = []
        if self.vary_channel:
            parts.append(f"信道{min(self.channels)}-{max(self.channels)}")
        if self.vary_power:
            parts.append(f"{self.power_min_db:.0f}~{self.power_max_db:.0f} dB")
        if self.vary_bw:
            parts.append("带宽" + "/".join(str(int(b)) for b in self.bw_choices_khz))
        return (f"每 {self.period_s:.1f} s 随机改 " + "、".join(parts)) if parts else "随机干扰"


@dataclass
class Action:
    channel_idx: int = 0
    power_db: float = 0.0
    bw_khz: float = 320.0
    waveform_mode: int = 0
    # 干扰块中心相对该信道中心的偏移；多信道压制时用来把块对齐到信道边界
    center_offset_hz: float = 0.0

    def clamped(self, cfg: "JamConfig") -> "Action":
        return Action(
            min(max(int(self.channel_idx), 0), cfg.num_channels - 1),
            min(max(float(self.power_db), POWER_MIN_DB), POWER_MAX_DB),
            float(self.bw_khz),
            int(self.waveform_mode) if int(self.waveform_mode) in (0, 1) else 0,
            float(self.center_offset_hz),
        )


@dataclass(frozen=True)
class JamConfig:
    # 频段
    center_freq_hz: float = 3.0e9
    num_channels: int = 10
    channel_width_hz: float = 160e3

    # 硬件
    ip_addr: str = "192.168.10.2"
    master_clock_rate: float = 200e6
    rx_ch: int = 0
    rx_antenna: str = "RX2"
    tx_ch: int = 0            # 与感知同板：先收后发
    tx_antenna: str = "TX/RX"
    victim_ch: int = 1        # 子板2：模拟通信信号
    victim_antenna: str = "TX/RX"
    rx_gain: float = 10.0
    tx_gain_fixed: float = 12.0
    # True：TX 本振跟着目标信道走，基带打在 DC（频谱上能量条就在该信道上）；
    # False：TX 本振固定在 center_freq_hz，靠基带搬移 ±(k-6)*160 kHz 对准信道。
    tx_lo_track: bool = True
    # 响应式跟踪的反应延迟：跟踪器判定切到新信道后，等这么久才真正把干扰搬过去
    # （等待期间继续压旧信道）。0 = 立即跟随。用于调试时把"感知→施扰"拉开。
    react_delay_s: float = 0.0
    # TDD 本振跳变：发射前把 RX/TX 本振一起搬到目标信道，发完立刻搬回感知中心。
    # 适用于 RX/TX 共用本振的板子 —— 这样除了感知窗口，目标信道以外不会有发射，
    # 3.000000 GHz 上也就不会常驻本振泄漏。代价：每帧两次调谐（长帧下可忽略）。
    lo_hop: bool = False
    # 空闲时把 TX 本振挪出观测频段：待机状态下 TX 本振若停在 3.000000，
    # 泄漏谱线正好落在观测带中心，看上去像"没发也有强线"。
    # 仅在 RX/TX 本振独立时生效（启动时自动探测）。
    park_tx_when_idle: bool = True
    park_tx_offset_hz: float = 20e6
    tx_peak_max: float = 0.85       # 20 dB 对应的基带峰值（0.85 ≈ -1.4 dBFS）
    tx_peak_min: float = 0.085      # 0 dB 对应（-21 dBFS）
    # 相邻信道覆盖数：>1 时把干扰块对齐到连续的 N 个信道（中心落在块中点，
    # 带宽 = N×160 kHz），用于"一次压住相邻几个信道"。1 = 只压目标信道。
    span_channels: int = 1
    sense_warmup_frames: int = 24   # 接收流启动瞬态约 10 帧，丢弃

    # 采样
    sense_rate: float = 2e6
    sense_samples: int = 4096
    tx_baseband_peak: float = 0.25

    # TDD：一帧干扰的时长。20 ms 太短，频谱仪上看是脉冲串；
    # 0.4 s 时占空比≈94%，看起来是连续压制（跟踪会变慢，需要快反应用 --jam-ms 20）
    jam_on_s: float = 0.4
    sense_gap_s: float = 0.0

    # 感知
    dc_notch_bins: int = 3
    dc_notch_hz: float = 40e3
    occ_snr_db: float = 12.0
    occ_peak_db: float = 20.0
    occ_contrast_db: float = 6.0
    wideband_snr_db: float = 16.0   # 相对绝对噪声基准，超过即判为宽带占用
    noise_ref_power: float = 0.0    # 0=自动学习；也可手动填线性功率

    # 跟踪
    acquire_frames: int = 2
    hop_confirm_frames: int = 2
    release_frames: int = 8
    on_threshold: float = 0.55
    off_threshold: float = 0.30
    hop_margin: float = 0.10

    # 中心频率漂移搜索（单位：1.6 MHz 块）
    scan_blocks: tuple[int, ...] = (0, 1, -1, 2, -2, 3, -3, 4, -4)
    scan_dwell_frames: int = 4
    scan_interval_frames: int = 40

    @property
    def channel_centers_hz(self) -> np.ndarray:
        return self.centers_at(self.center_freq_hz)

    def centers_at(self, center_hz: float) -> np.ndarray:
        k = np.arange(1, self.num_channels + 1) - 6
        return center_hz + k * self.channel_width_hz

    @property
    def jam_block_samples(self) -> int:
        return int(round(self.jam_on_s * self.sense_rate))

    @property
    def scan_offsets_hz(self) -> tuple[float, ...]:
        block = self.num_channels * self.channel_width_hz
        return tuple(b * block for b in self.scan_blocks)
