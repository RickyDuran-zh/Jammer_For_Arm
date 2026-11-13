"""目标跟踪状态机：发现占用 → 锁定 → 检测跳频 → 跟随 → 释放。

设计要点
- 迟滞双阈值：进入 0.55 / 退出 0.30，避免在判决边界抖动
- 连续帧确认：新目标需连续 acquire_frames 帧，跳频需 hop_confirm_frames 帧
- 自干扰遮蔽：本机干扰覆盖的信道不参与候选（全双工模式下必需）
- 跳频判据：旧目标掉到 off 阈值以下，且新目标分数显著更高（hop_margin）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class State(str, Enum):
    IDLE = "idle"          # 无目标，搜索中
    LOCKED = "locked"      # 已锁定并施加干扰


@dataclass
class Event:
    kind: str              # acquired / hop / lost / rejected
    frame: int
    channel: int | None = None      # 1-based
    prev_channel: int | None = None
    score: float = 0.0
    detail: str = ""


@dataclass
class TrackerPolicy:
    acquire_frames: int = 2
    hop_confirm_frames: int = 2
    release_frames: int = 8
    on_threshold: float = 0.55
    off_threshold: float = 0.30
    hop_margin: float = 0.10


@dataclass
class Tracker:
    num_channels: int
    policy: TrackerPolicy = field(default_factory=TrackerPolicy)

    target: int | None = None                 # 0-based
    state: State = State.IDLE
    frame: int = 0
    hop_count: int = 0
    last_scores: np.ndarray = field(default=None)
    events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._on_streak = np.zeros(self.num_channels, dtype=int)
        self._off_streak = np.zeros(self.num_channels, dtype=int)
        self._lost_frames = 0
        self.last_scores = np.zeros(self.num_channels)

    # ── 手动指定目标（Agent 或操作员）────────────────────────────
    def force_target(self, channel: int | None, reason: str = "manual") -> Event | None:
        prev = self.target
        if channel is None:
            if prev is None:
                return None
            self.target, self.state = None, State.IDLE
            self._reset_counters()
            ev = Event("lost", self.frame, None, prev + 1 if prev is not None else None,
                       detail=reason)
            self.events.append(ev)
            return ev
        ch = min(max(int(channel), 0), self.num_channels - 1)
        if prev == ch:                 # 目标没变：不产生事件，避免"3→3"的假跟随
            self.state = State.LOCKED
            self._lost_frames = 0
            self._reset_counters()
            return None
        self.target, self.state = ch, State.LOCKED
        self._lost_frames = 0
        self._reset_counters()
        ev = Event("acquired" if prev is None else "hop", self.frame, ch + 1,
                   prev + 1 if prev is not None else None,
                   float(self.last_scores[ch]), reason)
        self.events.append(ev)
        return ev

    # ── 每帧更新 ────────────────────────────────────────────────
    def update(self, scores: np.ndarray, masked: set[int] | None = None) -> list[Event]:
        self.frame += 1
        s = np.asarray(scores, dtype=float).copy()
        if masked:
            for k in masked:
                s[k] = -1.0
        self.last_scores = s
        events: list[Event] = []

        on = s >= self.policy.on_threshold
        off = s <= self.policy.off_threshold
        self._on_streak = np.where(on, self._on_streak + 1, 0)
        self._off_streak = np.where(off, self._off_streak + 1, 0)

        if self.target is None:
            ready = self._on_streak >= self.policy.acquire_frames
            if ready.any():
                cand = int(np.argmax(np.where(ready, s, -1)))
                self.target, self.state = cand, State.LOCKED
                self._lost_frames = 0
                ev = Event("acquired", self.frame, cand + 1, None, float(s[cand]))
                self.events.append(ev)
                events.append(ev)
            return events

        # 已锁定
        t = self.target
        if s[t] > self.policy.off_threshold:
            self._lost_frames = 0
            return events

        self._lost_frames += 1
        if self._lost_frames < self.policy.hop_confirm_frames:
            return events

        # 找新目标：必须持续 on，且比旧目标明显更强
        ready = (self._on_streak >= self.policy.hop_confirm_frames) & (s >= self.policy.on_threshold)
        ready[t] = False
        if ready.any():
            cand = int(np.argmax(np.where(ready, s, -1)))
            if s[cand] >= s[t] + self.policy.hop_margin:
                prev, self.target = t, cand
                self.hop_count += 1
                self._lost_frames = 0
                self._reset_counters()
                ev = Event("hop", self.frame, cand + 1, prev + 1,
                           float(s[cand]), f"score {s[prev]:.2f}->{s[cand]:.2f}")
                self.events.append(ev)
                events.append(ev)
                return events

        if self._lost_frames >= self.policy.release_frames:
            lost = self._lost_frames
            self.target, self.state = None, State.IDLE
            self._reset_counters()
            ev = Event("lost", self.frame, None, t + 1, float(s[t]),
                       f"no candidate for {lost} frames")
            self.events.append(ev)
            events.append(ev)
        return events

    def _reset_counters(self) -> None:
        self._on_streak[:] = 0
        self._off_streak[:] = 0
        self._lost_frames = 0
