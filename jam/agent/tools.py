"""Agent 工具：把自然语言意图映射到控制器原语。"""
from __future__ import annotations

import json

from ..config import BW_CHOICES_KHZ, POWER_MAX_DB, POWER_MIN_DB, WAVEFORM_NAMES
from ..controller import JamController

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "start_tracking",
            "description": "开始跟踪并干扰目标信道。不指定信道时自动搜索占用信道并锁定；"
                           "指定信道则直接锁定该信道，之后仍会自动跟随目标跳频。",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "integer", "description": "初始信道 1-10，省略则自动搜索"},
                    "waveform_mode": {"type": "string", "enum": ["noise", "multitone"],
                                      "description": "noise=限带噪声，multitone=多音"},
                    "power_db": {"type": "number", "description": "发射功率 0-20 dB"},
                    "bw_khz": {"type": "number", "enum": list(BW_CHOICES_KHZ),
                               "description": "干扰带宽 kHz"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_tracking",
            "description": "停止跟踪并停止发射干扰。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_center_frequency",
            "description": "切换干扰机的中心频率（MHz）。整段 1.6 MHz 观测窗口随中心移动，"
                           "切换后会清空当前目标并重新搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "freq_mhz": {"type": "number",
                                 "description": "中心频率 MHz，例如 3000 或 3010.5"},
                },
                "required": ["freq_mhz"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_jam_params",
            "description": "在不改变跟踪目标的前提下调整干扰参数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "waveform_mode": {"type": "string", "enum": ["noise", "multitone"]},
                    "power_db": {"type": "number"},
                    "bw_khz": {"type": "number", "enum": list(BW_CHOICES_KHZ)},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "查询当前跟踪状态：锁定信道、是否已发生跳频跟随、干扰参数、各信道占用评分。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_channel_features",
            "description": "查询 10 个信道的占用度、干扰水平、效用与类信噪比。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class ToolBox:
    def __init__(self, controller: JamController):
        self.c = controller
        self.handlers = {
            "start_tracking": self.start_tracking,
            "stop_tracking": self.stop_tracking,
            "set_center_frequency": self.set_center_frequency,
            "set_jam_params": self.set_jam_params,
            "get_status": self.get_status,
            "get_channel_features": self.get_channel_features,
        }

    # ── 工具实现 ──────────────────────────────────────────────
    def start_tracking(self, channel=None, waveform_mode=None, power_db=None,
                       bw_khz=None) -> str:
        mode = _mode(waveform_mode)
        if channel is not None and not 1 <= int(channel) <= 10:
            return "信道必须在 1-10 之间"
        if power_db is not None and not POWER_MIN_DB <= float(power_db) <= POWER_MAX_DB:
            return "功率必须在 0-20 dB 之间"
        if bw_khz is not None and float(bw_khz) not in BW_CHOICES_KHZ:
            return "带宽必须是 160/320/480/640/1600 kHz"

        self.c.start_tracking(int(channel) if channel is not None else None,
                              mode, power_db, bw_khz)
        what = f"信道 {channel}" if channel is not None else "自动搜索占用信道"
        extra = []
        if mode is not None:
            extra.append(WAVEFORM_NAMES[mode])
        if power_db is not None:
            extra.append(f"{float(power_db):.0f} dB")
        if bw_khz is not None:
            extra.append(f"{float(bw_khz):.0f} kHz")
        return f"已开始跟踪：{what}" + ("，" + "，".join(extra) if extra else "")

    def stop_tracking(self, **_):
        self.c.stop_tracking()
        return "已停止跟踪与干扰"

    def set_center_frequency(self, freq_mhz: float) -> str:
        f = float(freq_mhz)
        if not 300.0 <= f <= 6000.0:
            return "中心频率必须在 300-6000 MHz 之间"
        self.c.set_center(f * 1e6)
        return f"中心频率已切换到 {f:.3f} MHz"

    def set_jam_params(self, waveform_mode=None, power_db=None, bw_khz=None) -> str:
        mode = _mode(waveform_mode)
        if power_db is not None and not POWER_MIN_DB <= float(power_db) <= POWER_MAX_DB:
            return "功率必须在 0-20 dB 之间"
        if bw_khz is not None and float(bw_khz) not in BW_CHOICES_KHZ:
            return "带宽必须是 160/320/480/640/1600 kHz"
        self.c.set_params(mode, power_db, bw_khz)
        return "参数已更新"

    def get_status(self, **_) -> str:
        return json.dumps(self.c.status(), ensure_ascii=False)

    def get_channel_features(self, **_) -> str:
        res = self.c.latest
        if res is None:
            return json.dumps({"error": "暂无感知数据"}, ensure_ascii=False)
        rows = []
        for k in range(len(res.score)):
            rows.append({
                "信道": k + 1,
                "评分": round(float(res.score[k]), 3),
                "占用度": round(float(res.features[k, 0]), 3),
                "干扰水平": round(float(res.features[k, 1]), 3),
                "能量dB": round(float(res.energy_db[k]), 1),
                "状态": int(res.channel_map[k]),
            })
        return json.dumps(rows, ensure_ascii=False)

    def dispatch(self, name: str, args: dict) -> str:
        fn = self.handlers.get(name)
        if fn is None:
            return f"未知工具 {name}"
        return fn(**args)


def _mode(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.lower()
        if v in ("noise", "限带噪声", "噪声", "0"):
            return 0
        if v in ("multitone", "多音", "1"):
            return 1
        return None
    return 1 if int(value) == 1 else 0
