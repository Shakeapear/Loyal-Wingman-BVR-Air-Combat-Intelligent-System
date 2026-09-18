# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/llm_client.py（步骤 2.5 组件：LLM 接入层）
================================================================
统一封装三种 provider（stdlib urllib 实现，不引入新依赖）：

- deepseek：DeepSeek 开放平台（OpenAI 兼容协议），默认模型 deepseek-chat，
  API Key 取环境变量 DEEPSEEK_API_KEY；
- openai  ：OpenAI 兼容端点，默认模型 gpt-4o，API Key 取 OPENAI_API_KEY；
- mock    ：离线确定性模式（mock_llm.py），无需 Key，用于开发/单测/演示。

provider="auto" 时：有 DEEPSEEK_API_KEY 用 deepseek，否则回落 mock 并提示。
模型/密钥也可经构造参数或 CLI 显式指定；base_url 可用环境变量覆盖
（DEEPSEEK_BASE_URL / OPENAI_BASE_URL），便于走代理或本地兼容服务。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from . import mock_llm

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "base_env": "DEEPSEEK_BASE_URL",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "key_env": "OPENAI_API_KEY",
        "base_env": "OPENAI_BASE_URL",
    },
}


class LLMConfigError(RuntimeError):
    """LLM 配置缺失（如 API Key 未设置）。"""


def resolve_provider(provider: str) -> str:
    """auto → 有 DEEPSEEK_API_KEY 用 deepseek，否则 mock。"""
    if provider != "auto":
        return provider
    return "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "mock"


class LLMClient:
    def __init__(self, provider: str = "auto", model: str | None = None,
                 api_key: str | None = None, timeout: float = 90.0,
                 temperature: float = 0.2, max_retries: int = 2):
        self.provider = resolve_provider(provider)
        if self.provider not in (*PROVIDERS, "mock"):
            raise LLMConfigError(f"未知 provider: {provider!r}（可选 auto/deepseek/openai/mock）")
        spec = PROVIDERS.get(self.provider, {})
        self.model = model or spec.get("model")
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.base_url = os.environ.get(spec.get("base_env", ""), spec.get("base_url", ""))
        if self.provider != "mock":
            self.api_key = api_key or os.environ.get(spec["key_env"])
            if not self.api_key:
                raise LLMConfigError(
                    f"provider={self.provider} 需要 API Key：请设置环境变量 "
                    f"{spec['key_env']}（或使用 --provider mock 离线运行）")

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def complete(self, system: str, user: str, task: str | None = None,
                 context: dict | None = None) -> str:
        """调用 LLM（mock 走确定性实现），返回原始文本。"""
        if self.is_mock:
            return mock_llm.respond(task or "", context or {})
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "stream": False,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败（provider={self.provider}, url={url}）: {last_err}")


def extract_json(text: str):
    """从 LLM 输出中提取 JSON（容忍 ```json 围栏与前后说明文字）。"""
    if not isinstance(text, str):
        raise ValueError(f"LLM 输出应为文本，实际 {type(text)!r}")
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM 输出中未找到 JSON 对象")
    depth, in_str, esc, end = 0, False, False, None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise ValueError("LLM 输出中的 JSON 大括号不配对")
    return json.loads(text[start:end + 1])
