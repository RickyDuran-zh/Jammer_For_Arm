"""Agent 工具：把自然语言意图映射到控制器原语。"""
from __future__ import annotations

import json
import time

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
            "name": "start_fixed_jamming",
            "description": "在指定信道上固定干扰：锁死该信道，不跟随对方切频，也不会自动释放。"
                           "用户说“固定干扰/锁定在/就压着/不要跟着跳”时用这个，"
                           "不要用 start_tracking。",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "integer", "description": "信道 1-10（必填）"},
                    "waveform_mode": {"type": "string", "enum": ["noise", "multitone"],
                                      "description": "noise=限带噪声，multitone=多音"},
                    "power_db": {"type": "number", "description": "发射功率 0-20 dB"},
                    "bw_khz": {"type": "number", "enum": list(BW_CHOICES_KHZ),
                               "description": "干扰带宽 kHz"},
                },
                "required": ["channel"],
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
            "name": "analyze_situation",
            "description": "分析当前频谱态势：目标信道、各信道占用/干扰状态、干扰参数、"
                           "是否稳定，并给出判断和建议。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend_history",
            "description": "查看最近若干帧的运行趋势：全局 SNR、最高信道评分、目标信道。",
            "parameters": {
                "type": "object",
                "properties": {
                    "frames": {"type": "integer",
                               "description": "回看帧数，默认 60，最大 900"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_channel_map_history",
            "description": "查看最近若干帧的 10 信道状态历史（空/占/扰/叠）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "frames": {"type": "integer",
                               "description": "回看帧数，默认 20，最大 300"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_system_info",
            "description": "查询系统配置与运行信息：中心频率、端口、TDD 时序、阈值、运行时长、发射次数。",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
            "start_fixed_jamming": self.start_fixed_jamming,
            "stop_tracking": self.stop_tracking,
            "set_center_frequency": self.set_center_frequency,
            "set_jam_params": self.set_jam_params,
            "analyze_situation": self.analyze_situation,
            "get_trend_history": self.get_trend_history,
            "get_channel_map_history": self.get_channel_map_history,
            "query_system_info": self.query_system_info,
            "get_status": self.get_status,
            "get_channel_features": self.get_channel_features,
        }

    # ── 工具实现 ──────────────────────────────────────────────
    def start_fixed_jamming(self, channel, waveform_mode=None, power_db=None,
                            bw_khz=None) -> str:
        ch = int(channel)
        if not 1 <= ch <= 10:
            return "信道必须在 1-10 之间"
        mode = _mode(waveform_mode) if waveform_mode is not None else None
        self.c.start_fixed(ch, waveform_mode=mode, power_db=power_db, bw_khz=bw_khz)
        return f"已在信道{ch}固定干扰（锁定，不跟随切频）。"

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

    def get_trend_history(self, frames: int = 60, **_) -> str:
        hist = self.c.trend_history(int(frames))
        if not hist:
            return "暂无趋势数据"
        step = max(1, len(hist) // 12)
        rows = hist[::step][-12:]
        lines = ["帧 | SNR(dB) | 最高评分 | 目标"]
        for f, s_, m, t in rows:
            lines.append(f"{f} | {s_:.1f} | {m:.2f} | {t or '-'}")
        snrs = [h[1] for h in hist]
        scores = [h[2] for h in hist]
        lines.append(f"区间：SNR {min(snrs):.1f}~{max(snrs):.1f} dB，"
                     f"最高评分 {min(scores):.2f}~{max(scores):.2f}")
        return "\n".join(lines)

    def get_channel_map_history(self, frames: int = 20, **_) -> str:
        hist = self.c.map_history(int(frames))
        if not hist:
            return "暂无信道历史"
        names = {0: "空", 1: "占", 2: "扰", 3: "叠"}
        lines = ["帧 | 1 2 3 4 5 6 7 8 9 10"]
        for i, row in enumerate(hist[-12:]):
            lines.append(f"{i + 1:>2} | " + " ".join(
                names.get(int(v), "?") for v in row))
        return "\n".join(lines)

    def query_system_info(self, **_) -> str:
        i = self.c.system_info()
        return "\n".join([
            f"中心 {i['center_mhz']:.3f} MHz ｜ {i['channels']} 信道 × "
            f"{i['channel_khz']:.0f} kHz ｜ 采样 {i['sample_rate_msps']:.1f} MS/s",
            f"端口：感知 {i['rx_port']} ｜ 干扰 {i['tx_port']} ｜ "
            f"模拟信号 {i['victim_port']}",
            f"TDD：干扰 {i['jam_ms']:.0f} ms ｜ 周期 {i['cycle_ms']:.1f} ms ｜ "
            f"TX 增益 {i['tx_gain_db']:.0f} dB",
            f"阈值 {i['thresholds'][0]:.2f}/{i['thresholds'][1]:.2f} ｜ "
            f"释放 {i['release_frames']} 帧",
            f"运行 {i['uptime_s']:.0f} 秒 ｜ 累计发射 {i['tx_frames']} 次 ｜ "
            f"模式 {i['mode']}",
        ])

    def analyze_situation(self, **_) -> str:
        st = self.c.status()
        res = self.c.latest
        mode = {"fixed": "固定干扰", "auto": "自动跟踪",
                "stopped": "已停止"}.get(st.get("mode"), "-")
        target = st.get("target_ch")
        if not st.get("tracking", True):
            state = "已停止"
        elif st.get("scanning"):
            state = "扫描中"
        elif target:
            state = "锁定"
        else:
            state = "搜索"

        lines = [f"【态势分析】{time.strftime('%H:%M:%S')}",
                 f"中心 {st.get('center_mhz', 0):.3f} MHz ｜ 模式 {mode} ｜ 状态 {state}"]
        if target:
            lat = st.get("hop_latency_ms")
            extra = f" ｜ 最近感知延迟 {lat:.0f} ms" if lat else ""
            lines.append(f"目标 信道{target} ｜ 已跟随 {st.get('hop_count', 0)} 次{extra}")
        else:
            lines.append(f"目标 无 ｜ 已跟随 {st.get('hop_count', 0)} 次")
        lines.append(f"全局 SNR {st.get('snr_est', 0):.1f} dB ｜ 周期 "
                     f"{st.get('cycle_ms', 0):.1f} ms ｜ 累计发射 {st.get('tx_frames', 0)} 次")

        if res is not None:
            names = {0: "空", 1: "占", 2: "扰", 3: "叠"}
            chs = " ".join(f"{k + 1}{names.get(int(res.channel_map[k]), '?')}"
                           for k in range(len(res.channel_map)))
            lines.append("信道 " + chs)
            order = sorted(range(len(res.score)), key=lambda k: res.score[k],
                           reverse=True)[:3]
            top = "、".join(f"信道{int(k) + 1}({res.score[k]:.2f})"
                            for k in order if res.score[k] > 0.2)
            if top:
                lines.append(f"最强占用：{top}")

        lines.append(f"干扰 {WAVEFORM_NAMES.get(st.get('waveform', 0), '-')} ｜ "
                     f"{st.get('bw_khz', 0):.0f} kHz ｜ {st.get('power_db', 0):.0f} dB")

        if not st.get("tracking", True):
            judge = "跟踪已停止，当前不发射干扰。"
        elif st.get("scanning"):
            judge = "正在扫描其他 1.6 MHz 频块，尚未锁定目标。"
        elif target:
            judge = f"已锁定信道{target}并持续压制，目标切频后会自动跟随。"
        else:
            judge = "当前窗口内未发现可用目标，继续搜索。"

        lat = st.get("hop_latency_ms") or 0
        if target and lat > 300:
            advice = "感知延迟偏大，可缩短干扰时长或提高控制帧率。"
        elif target:
            advice = "态势稳定，可维持当前参数。"
        elif st.get("scanning"):
            advice = "若长时间无目标，检查天线或扩大扫描范围。"
        else:
            advice = "已知目标频点时，可用固定干扰直接压制。"

        return "\n".join(lines + [f"判断：{judge}", f"建议：{advice}"])

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
