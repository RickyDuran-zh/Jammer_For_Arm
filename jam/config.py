"""全局配置：1.6 MHz 观测带宽、10×160 kHz 信道、双子板端口映射。

功率策略（现场 5 cm 天线间距、无衰减器，实测得出）：
  TX 增益固定 0 dB，干扰强度全部由基带幅度表达。
  实测满幅 0.25 时对端接收仅 ~0.006（无削顶），留足安全余量；
  因此不做 RF 增益调节，避免误操作把 UBX 前端压死。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BW_CHOICES_KHZ: tuple[float, ...] = (160.0, 320.0, 480.0, 640.0, 1600.0)
POWER_MIN_DB: float = 0.0
POWER_MAX_DB: float = 20.0
WAVEFORM_NAMES: dict[int, str] = {0: "限带噪声", 1: "多音"}

CH_IDLE, CH_USER, CH_JAMMED, CH_OVERLAP = 0, 1, 2, 3


@dataclass
class Action:
    channel_idx: int = 0
    power_db: float = 0.0
    bw_khz: float = 320.0
    waveform_mode: int = 0

    def clamped(self, cfg: "JamConfig") -> "Action":
        return Action(
            min(max(int(self.channel_idx), 0), cfg.num_channels - 1),
            min(max(float(self.power_db), POWER_MIN_DB), POWER_MAX_DB),
            float(self.bw_khz),
            int(self.waveform_mode) if int(self.waveform_mode) in (0, 1) else 0,
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
    tx_gain_fixed: float = 0.0
    tx_peak_max: float = 0.25       # 20 dB 对应的基带峰值
    tx_peak_min: float = 0.025      # 0 dB 对应
    sense_warmup_frames: int = 24   # 接收流启动瞬态约 10 帧，丢弃

    # 采样
    sense_rate: float = 2e6
    sense_samples: int = 4096
    tx_baseband_peak: float = 0.25

    # TDD
    jam_on_s: float = 0.020
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
