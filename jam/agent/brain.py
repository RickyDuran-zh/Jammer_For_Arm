"""Agent 编排：云端模型优先做语义理解，失败时用本地规则兜底。"""
from __future__ import annotations

import re

from ..controller import JamController
from .llm import DeepSeekClient, LLMError
from .tools import TOOL_DEFS, ToolBox

SYSTEM_PROMPT = """你是应急频谱管控干扰机的控制助手。系统会自动感知 3 GHz 附近 1.6 MHz 内的 10 个信道（每个 160 kHz），
锁定正在通信的信道并施加干扰；当通信方切频时，跟踪器会自动跟随到新信道。

你的职责：把操作员的一句话解析成一次或多次工具调用。

规则：
1. 凡是涉及控制（开始/停止跟踪、改功率、改带宽、换波形），必须调用工具，不得凭空回复"已完成"。
2. 一句话里包含多个参数时，合并进同一次工具调用。
3. 用户说"固定干扰/锁定在/就压着/不要跟着跳/别跟"→ start_fixed_jamming(channel=N)（锁死，不跟随）。
   用户说"随机干扰/随机时变/盲扫/乱压"→ start_random_jamming(period_s=N)（按周期随机换参数，不看对方动作）。
   用户说"延迟300毫秒/响应时间改成0.2秒/取消延迟"→ set_jam_params(react_delay_ms=N)（0 表示立即跟随）。
   用户说"跟踪/自动跟随/跟着它/压制某信道"→ start_tracking(channel=N)；说"自动找/自己找"→ start_tracking()。
   用户问"态势/分析一下/当前情况"→ analyze_situation()；
   问"趋势/最近怎么变"→ get_trend_history()；问"信道历史/流水"→ get_channel_map_history()；
   问"配置/系统信息/硬件"→ query_system_info()。
4. 用户没说参数就不要传，不要臆造默认值。
5. 回复中文、简短（80 字内），说明实际执行了什么。

参数范围：功率 -30~20 dB（负数表示减弱）；带宽 160/320/480/640/1600 kHz；波形 noise（限带噪声）/ multitone（多音）。
中心频率可用 set_center_frequency 切换，单位 MHz，例如 3000 或 3010.5。"""


class JamAgent:
    def __init__(self, controller: JamController, client: DeepSeekClient | None = None):
        self.controller = controller
        self.tools = ToolBox(controller)
        self.client = client or DeepSeekClient()
        self.history: list[dict] = []

    def handle(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "请输入指令。"

        if self.client.available:
            try:
                messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                            + self.history[-6:]
                            + [{"role": "user", "content": text}])
                reply, actions = self.client.chat(messages, TOOL_DEFS, self.tools.dispatch)
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": reply})
                self.history = self.history[-6:]
                if reply:
                    return reply
                return "已执行：" + "；".join(actions) if actions else "已执行。"
            except LLMError as e:
                print(f"[Agent] 云端不可用，改用本地解析：{e}")

        return self._fallback(text)

    # ── 本地兜底（断网/超时）─────────────────────────────────
    def _fallback(self, text: str) -> str:
        t = text.replace("，", ",").replace("。", ".")

        # 先处理"取消延迟"这类，否则会被下面的"取消"当成停止干扰
        if (any(k in t for k in ("延迟", "延时", "时延", "反应时间", "响应时间"))
                and any(k in t for k in ("取消", "关闭", "不要", "去掉", "清零", "立即"))):
            return self.tools.set_jam_params(react_delay_ms=0)

        if ("延迟" not in t and "时延" not in t
                and any(k in t for k in ("停止", "取消", "别干扰", "关闭干扰", "停掉"))):
            return self.tools.stop_tracking()

        if any(k in t for k in ("趋势", "最近怎么", "变化")):
            return self.tools.get_trend_history()

        if any(k in t for k in ("信道历史", "流水", "热力", "历史")):
            return self.tools.get_channel_map_history()

        if any(k in t for k in ("配置", "系统信息", "硬件", "版本", "运行多久")):
            return self.tools.query_system_info()

        if any(k in t for k in ("态势", "分析", "评估")):
            return self.tools.analyze_situation()

        if any(k in t for k in ("状态", "情况", "在哪", "跟踪到")):
            st = self.controller.status()
            target = st.get("target_ch")
            return (f"当前{'锁定信道 ' + str(target) if target else '未锁定'}，"
                    f"已跟随跳频 {st.get('hop_count', 0)} 次，"
                    f"功率 {st.get('power_db', 0):.0f} dB，"
                    f"波形 {'多音' if st.get('waveform') == 1 else '限带噪声'}。")

        m = re.search(r'(?:中心频率|中心|center)\s*(\d{3,4}(?:\.\d+)?)', t, re.I)
        if m:
            return self.tools.set_center_frequency(float(m.group(1)))

        mode = None
        if any(k in t for k in ("多音", "点频")):
            mode = 1
        elif any(k in t for k in ("噪声", "限带")):
            mode = 0

        m = re.search(r'(提高|增加|加|降低|减少|减)\s*(\d+(?:\.\d+)?)\s*(?:db|分贝)',
                      t, re.I)
        if m:
            cur = float(self.controller.status().get("power_db", 10))
            delta = float(m.group(2)) * (1 if m.group(1) in ("提高", "增加", "加") else -1)
            return self.tools.set_jam_params(power_db=min(20.0, max(0.0, cur + delta)))

        m = re.search(r'(?:延迟|延时|时延|反应时间|响应时间|响应延迟)'
                      r'[^\d]{0,6}(\d+(?:\.\d+)?)\s*(ms|毫秒|秒|s)?',
                      t, re.I)
        if m:
            val = float(m.group(1))
            if (m.group(2) or "ms").lower() in ("秒", "s"):
                val *= 1000.0
            return self.tools.set_jam_params(react_delay_ms=val)

        power = None
        m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:db|分贝)', t, re.I)
        if m:
            power = float(m.group(1))

        bw = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:khz|k)', t, re.I)
        if m:
            bw = float(m.group(1))

        ch = None
        m = re.search(r'(?:信道|通道|频道|ch)\s*(\d+)', t, re.I) or re.search(r'(\d+)\s*号', t)
        if m:
            ch = int(m.group(1))

        if any(k in t for k in ("切到", "换到", "切至")) and ch is not None:
            return self.tools.start_fixed_jamming(ch, waveform_mode=mode,
                                                  power_db=power, bw_khz=bw)

        if any(k in t for k in ("随机", "盲扫", "乱压", "时变")):
            per = 2.0
            m2 = re.search(r'(\d+(?:\.\d+)?)\s*秒', t)
            if m2:
                per = float(m2.group(1))
            return self.tools.start_random_jamming(period_s=per, waveform_mode=mode)

        start = any(k in t for k in ("跟踪", "干扰", "压制", "锁定", "打"))
        fixed = any(k in t for k in ("固定", "锁死", "不要跟", "别跟", "就压"))
        if start or fixed:
            if fixed and ch is not None:
                return self.tools.start_fixed_jamming(ch, waveform_mode=mode,
                                                      power_db=power, bw_khz=bw)
            return self.tools.start_tracking(channel=ch, waveform_mode=mode,
                                             power_db=power, bw_khz=bw)
        if any(v is not None for v in (mode, power, bw)):
            return self.tools.set_jam_params(waveform_mode=mode, power_db=power, bw_khz=bw)
        return "没听懂。可说：跟踪信道3、自动跟踪、随机干扰、固定干扰、功率15dB、换多音、停止干扰、查询状态。"
