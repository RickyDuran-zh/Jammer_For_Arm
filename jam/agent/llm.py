"""DeepSeek 云端语义理解客户端（函数调用）。

注意：语义理解只在"下一道指令"时调用，绝不进入实时跟踪环；
跟踪与发射全部是本地确定性逻辑，断网也能继续按当前策略工作。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# 优先读环境变量；本地开发可放在 gitignore 的 key_local.py 里
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    try:
        from .key_local import API_KEY as _LOCAL_KEY
        API_KEY = _LOCAL_KEY
    except Exception:
        pass
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"          # DeepSeek-V4.1-Flash
TIMEOUT_S = 60.0


class LLMError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, api_key: str = API_KEY, model: str = MODEL,
                 base_url: str = BASE_URL):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, messages: list[dict], tools: list[dict] | None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 800,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}") from e
        except Exception as e:
            raise LLMError(f"请求失败: {e}") from e

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             dispatch=None, max_rounds: int = 3) -> tuple[str, list[str]]:
        """返回 (回复文本, 已执行动作描述列表)。"""
        msgs = list(messages)
        actions: list[str] = []
        text = ""

        for _ in range(max_rounds):
            resp = self._post(msgs, tools)
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            calls = msg.get("tool_calls") or []
            if not calls:
                return text, actions

            msgs.append({"role": "assistant", "content": text, "tool_calls": calls})
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = dispatch(name, args) if dispatch else "未注册工具"
                except Exception as e:
                    result = f"执行失败: {e}"
                actions.append(f"{name}({json.dumps(args, ensure_ascii=False)}) -> {result}")
                msgs.append({"role": "tool", "tool_call_id": call["id"],
                             "content": str(result)})

        return text or "已执行指令。", actions
