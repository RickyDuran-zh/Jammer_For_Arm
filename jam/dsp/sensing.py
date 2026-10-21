"""感知：FFT → 10 信道四维特征 → 目标评分。

channel_centers 由当前中心频率决定：中心频率可变，信道宽度固定 160 kHz。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CH_JAMMED, CH_OVERLAP, CH_USER, Action, JamConfig


@dataclass
class SensingResult:
    features: np.ndarray
    score: np.ndarray
    energy_db: np.ndarray
    channel_map: np.ndarray
    snr_est: float
    bw_est: float
    sig_type: int
    noise_floor_db: float = 0.0


def _notch_half_bins(n: int, cfg: JamConfig) -> int:
    half = 0
    if cfg.dc_notch_bins > 0:
        half = max(half, cfg.dc_notch_bins // 2)
    if cfg.dc_notch_hz > 0:
        half = max(half, int(np.ceil(cfg.dc_notch_hz / 2 / (cfg.sense_rate / n))))
    return half


def suppress_center_dc(x: np.ndarray, cfg: JamConfig) -> np.ndarray:
    y = x.copy()
    n = y.size
    half = _notch_half_bins(n, cfg)
    if half <= 0:
        return y
    c = n // 2
    lo, hi = max(0, c - half), min(n - 1, c + half)
    pad = max(20, half)
    idx = np.arange(max(0, lo - 2 * pad), min(n - 1, hi + 2 * pad) + 1)
    keep = (idx < lo) | (idx > hi)
    ref = y[idx[keep]]
    y[lo:hi + 1] = float(np.median(ref)) if ref.size else float(np.median(y))
    return y


def _affected_channels(action: Action, cfg: JamConfig, center_hz: float) -> list[int]:
    centers = cfg.centers_at(center_hz)
    bw_hz = float(action.bw_khz) * 1e3
    center = centers[action.channel_idx]
    left, right = center - bw_hz / 2, center + bw_hz / 2
    out = []
    for k in range(cfg.num_channels):
        ch_l = centers[k] - cfg.channel_width_hz / 2
        ch_r = centers[k] + cfg.channel_width_hz / 2
        if ch_r > left and ch_l < right:
            out.append(k)
    return out or [action.channel_idx]


def covered_channels(action: Action | None, cfg: JamConfig,
                     center_hz: float | None = None) -> set[int]:
    if action is None:
        return set()
    return set(_affected_channels(action, cfg, center_hz or cfg.center_freq_hz))


def sense(x: np.ndarray, cfg: JamConfig, prev_action: Action | None = None,
          center_hz: float | None = None,
          noise_ref_power: float = 0.0) -> SensingResult:
    center_hz = center_hz or cfg.center_freq_hz
    centers = cfg.centers_at(center_hz)
    fs, n_ch = cfg.sense_rate, cfg.num_channels
    n = max(4096, int(x.size))

    sig = np.asarray(x, dtype=np.complex128).ravel()
    sig = sig - sig.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft(sig, n))) ** 2
    power = suppress_center_dc(power, cfg)
    pnorm = power / (power.max() + 1e-12)
    if noise_ref_power and noise_ref_power > 0:
        # 用绝对噪声基准（安静频段学到的），宽带占用才不会被当成底噪
        noise_floor = float(noise_ref_power) * n
    else:
        noise_floor = float(np.median(power)) + 1e-12

    strong = pnorm > 0.3
    weak = pnorm < 0.3
    p_sig = float(power[strong].mean()) if strong.sum() >= 10 else float(power.mean())
    p_noi = float(power[weak].mean()) if weak.sum() >= 10 else float(power.mean())
    snr_est = float(10 * np.log10(p_sig / (p_noi + 1e-12)))

    active = np.flatnonzero(pnorm > 0.1)
    if active.size == 0:
        bw_est, sig_type = 0.0, 0
    else:
        bw_est = float(active.max() - active.min()) * (fs / n)
        sig_type = 1 if bw_est <= cfg.channel_width_hz else (
            2 if bw_est <= 4 * cfg.channel_width_hz else 3)

    freqs = center_hz + (np.arange(n) - n / 2) * (fs / n)
    features = np.zeros((n_ch, 4))
    energy_db = np.full(n_ch, -np.inf)
    peak_db = np.full(n_ch, -np.inf)
    snr_db_vec = np.full(n_ch, -np.inf)
    channel_map = np.zeros(n_ch, dtype=np.int8)

    for k in range(n_ch):
        center = centers[k]
        idx = np.flatnonzero(np.abs(freqs - center) < cfg.channel_width_hz / 2)
        if cfg.dc_notch_hz > 0:
            idx = idx[np.abs(freqs[idx] - center_hz) > cfg.dc_notch_hz / 2]
        if idx.size == 0:
            continue
        e = float(power[idx].mean())
        p = float(power[idx].max())
        snr_db = 10 * np.log10(e / noise_floor)
        peak = 10 * np.log10(p / noise_floor)
        energy_db[k] = 10 * np.log10(e + 1e-12)
        peak_db[k] = peak
        snr_db_vec[k] = snr_db

        occ = float(np.clip((snr_db - cfg.occ_snr_db) / 14, 0, 1))
        interf = float(np.clip((peak - 12) / 18, 0, 1))
        util = float(np.clip(0.65 * occ + 0.35 * max(0.0, peak / 25) - 0.25 * interf, 0, 1))
        snr_like = float(np.clip(snr_db / 25, 0, 1))
        features[k] = (occ, interf, util, snr_like)

    valid = energy_db[np.isfinite(energy_db)]
    bg = float(np.median(valid)) if valid.size else 10 * np.log10(noise_floor)

    for k in range(n_ch):
        if not np.isfinite(energy_db[k]):
            continue
        lo, hi = max(0, k - 2), min(n_ch, k + 3)
        neigh = [i for i in range(lo, hi) if i != k and np.isfinite(energy_db[i])]
        nb = float(np.median(energy_db[neigh])) if neigh else bg
        contrast = energy_db[k] - max(bg, nb)
        occ_contrast = float(np.clip((contrast - cfg.occ_contrast_db) / 12, 0, 1))
        occ_wide = float(np.clip((snr_db_vec[k] - cfg.wideband_snr_db + 6) / 12, 0, 1))
        features[k, 0] = max(occ_contrast, occ_wide)
        narrowband = (peak_db[k] >= cfg.occ_peak_db and contrast >= cfg.occ_contrast_db)
        wideband = snr_db_vec[k] >= cfg.wideband_snr_db
        if features[k, 3] * 25 >= cfg.occ_snr_db and (narrowband or wideband):
            channel_map[k] = CH_USER

    if prev_action is not None and float(prev_action.power_db) > 0:
        for k in _affected_channels(prev_action, cfg, center_hz):
            channel_map[k] = CH_OVERLAP if channel_map[k] in (CH_USER, CH_OVERLAP) else CH_JAMMED

    score = np.zeros(n_ch)
    floor_db = 10 * np.log10(noise_floor)
    for k in range(n_ch):
        if not np.isfinite(energy_db[k]):
            continue
        # 占用度在高 SNR 下会饱和，加入不饱和的强度项来区分强弱用户
        strength = float(np.clip((energy_db[k] - floor_db - 8) / 32, 0, 1))
        score[k] = float(np.clip(0.55 * features[k, 0] + 0.20 * features[k, 3]
                                 + 0.25 * strength, 0, 1))

    return SensingResult(features, score, energy_db, channel_map, snr_est,
                         bw_est, sig_type, floor_db)
