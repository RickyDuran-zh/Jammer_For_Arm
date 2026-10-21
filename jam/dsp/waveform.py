"""干扰波形：mode 0 限带噪声、mode 1 多音。

功率映射：TX 增益固定 0 dB，power_db 0~20 → 基带峰值 0.025~0.25。
现场 5 cm 间距下满幅也不会让对端削顶（实测），因此不再动 RF 增益。
"""
from __future__ import annotations

import numpy as np

from ..config import Action, JamConfig


def _noise(n: int, fs: float, f0: float, bw_hz: float,
           rng: np.random.Generator) -> np.ndarray:
    freq = (np.arange(n) - n / 2) * (fs / n)
    in_band = np.abs(freq - f0) <= bw_hz / 2
    spec = np.zeros(n, dtype=np.complex128)
    count = int(in_band.sum())
    spec[in_band] = rng.standard_normal(count) + 1j * rng.standard_normal(count)
    return np.fft.ifft(np.fft.ifftshift(spec))


def _multitone(n: int, fs: float, f0: float, bw_hz: float,
               rng: np.random.Generator) -> np.ndarray:
    tones = 4
    t = np.arange(n) / fs
    offsets = np.linspace(-0.375 * bw_hz, 0.375 * bw_hz, tones)
    y = np.zeros(n, dtype=np.complex128)
    for off in offsets:
        y += np.exp(1j * 2 * np.pi * (f0 + off) * t)
    return y / np.sqrt(tones)


_GENERATORS = {0: _noise, 1: _multitone}


def generate(action: Action, cfg: JamConfig, length: int | None = None,
             rng: np.random.Generator | None = None,
             center_hz: float | None = None) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    a = action.clamped(cfg)
    n = int(length or cfg.sense_samples)
    center = center_hz or cfg.center_freq_hz
    f0 = cfg.centers_at(center)[a.channel_idx] - center

    amp = cfg.tx_peak_min * (10 ** (a.power_db / 20))
    gen = _GENERATORS.get(a.waveform_mode, _noise)
    y = gen(n, cfg.sense_rate, f0, a.bw_khz * 1e3, rng)

    peak = float(np.abs(y).max())
    if peak < 1e-12:
        return np.zeros(n, dtype=np.complex128), cfg.tx_gain_fixed
    return amp * y / peak, cfg.tx_gain_fixed
