"""联调记录：把状态快照和跟踪事件追加成 JSONL。

只在一个独立线程里读 controller.status()，不进入实时环、不参与任何决策，
因此开不开记录都不影响 TDD 节拍。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

STATUS_FIELDS = (
    "frames", "state", "mode", "jam_mode", "jam_mode_name",
    "random_period_s", "random_next_s", "random_last",
    "target_ch", "hop_count", "hop_latency_ms",
    "cycle_ms", "jam_on_ms", "duty_pct", "center_mhz", "tx_freq_mhz", "tx_lo_track", "tx_tune_ms",
    "tx_err_hz", "lo_hop", "lo_independent", "tx_parked", "react_delay_ms", "react_active", "react_pending_ms",
    "last_react_ms", "jam_target_ch", "span_channels", "span_to_ch", "bw_set_khz",
    "power_db", "bw_khz", "waveform", "tx_frames",
    "snr_est", "noise_ref", "scores", "channel_map", "victim_on",
)


class StatusLogger:
    """每 period_s 秒写一行 status，事件发生时额外写一行 event。"""

    def __init__(self, controller, path, period_s: float = 0.2, case: str = ""):
        self.controller = controller
        self.path = Path(path)
        self.period_s = float(period_s)
        self.case = case
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._fh = None

    def start(self) -> "StatusLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        cfg = self.controller.cfg
        self._write({
            "t": round(time.time(), 3),
            "kind": "meta",
            "case": self.case,
            "center_mhz": round(cfg.center_freq_hz / 1e6, 4),
            "channels": cfg.num_channels,
            "channel_khz": cfg.channel_width_hz / 1e3,
            "jam_ms": cfg.jam_on_s * 1e3,
        })
        self._thread = threading.Thread(target=self._loop, name="status-log", daemon=True)
        self._thread.start()
        return self

    def _write(self, obj: dict) -> None:
        with self._lock:
            if self._fh is None:
                return
            try:
                self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                self._fh.flush()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                st = self.controller.status()
                row = {"t": round(time.time(), 3), "kind": "status"}
                for k in STATUS_FIELDS:
                    row[k] = st.get(k)
                row["error"] = st.get("error")
                self._write(row)
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def event(self, ev) -> None:
        try:
            self._write({
                "t": round(time.time(), 3),
                "kind": "event",
                "event": ev.kind,
                "frame": ev.frame,
                "channel": ev.channel,
                "prev_channel": ev.prev_channel,
                "score": round(float(ev.score), 3),
                "detail": ev.detail,
            })
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def attach(controller, path, case: str = "", period_s: float = 0.2):
    """挂上记录器，同时保留原有的 on_event 回调。返回 logger 或 None。"""
    try:
        logger = StatusLogger(controller, path, period_s=period_s, case=case).start()
    except Exception as exc:                     # 记录失败不能拖垮主流程
        print(f"[记录] 无法写入 {path}: {exc}")
        return None
    prev = controller.on_event

    def sink(ev):
        if prev is not None:
            try:
                prev(ev)
            except Exception:
                pass
        logger.event(ev)

    controller.on_event = sink
    return logger
