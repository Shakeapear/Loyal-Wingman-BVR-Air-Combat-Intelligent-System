# -*- coding: utf-8 -*-
"""
多任务智能体/common/errors.py（统一异常与错误码，接口规范 §2.3）
================================================================
层级：ProjectError(RuntimeError) → ConfigError / ValidationError /
      DataFormatError / ExecutionError / ExternalServiceError

约定：
- 每个异常带稳定 `code`（字符串常量，见本模块顶部列表）与 `detail`（字段路径/上下文）；
- 兼容旧调用方：ProjectError 是 RuntimeError 子类；ConfigError/ValidationError
  同时是 ValueError 子类（历史代码/测试中的 except ValueError 仍然生效）；
- CLI 统一用 `exit_code_for(exc)` 映射退出码：0 成功 / 1 业务失败 /
  2 参数或配置错误 / 3 外部服务失败。
"""
from __future__ import annotations

# ---- 错误码表（前缀化：CFG/ARG/DATA/ENV/BT/LLM/IO）----
CFG_INVALID = "CFG_INVALID"
ARG_INVALID = "ARG_INVALID"
DATA_BAD_COLUMN = "DATA_BAD_COLUMN"
DATA_BAD_VERSION = "DATA_BAD_VERSION"
DATA_MISSING_FIELD = "DATA_MISSING_FIELD"
ENV_NOT_RESET = "ENV_NOT_RESET"
ENV_FIRE_FAILED = "ENV_FIRE_FAILED"
BT_SCHEMA_INVALID = "BT_SCHEMA_INVALID"
BT_BUILD_FAILED = "BT_BUILD_FAILED"
DEP_MISSING = "DEP_MISSING"
LLM_AUTH_MISSING = "LLM_AUTH_MISSING"
LLM_HTTP_ERROR = "LLM_HTTP_ERROR"
IO_PATH_DENIED = "IO_PATH_DENIED"

EXIT_OK = 0
EXIT_BIZ = 1
EXIT_CONFIG = 2
EXIT_EXTERNAL = 3


class ProjectError(RuntimeError):
    """项目异常基类（RuntimeError 子类，兼容旧 except RuntimeError）。"""

    def __init__(self, code: str, message: str, detail=None, cause=None):
        self.code = code
        self.message = message
        self.detail = detail
        self.cause = cause
        text = f"[{code}] {message}"
        if detail:
            text += f" | {detail}"
        super().__init__(text)

    def to_dict(self):
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigError(ProjectError, ValueError):
    """配置/参数取值非法（兼容旧 except ValueError）。"""


class ValidationError(ProjectError, ValueError):
    """入参或数据契约校验失败（兼容旧 except ValueError）。"""


class DataFormatError(ProjectError):
    """数据文件格式/版本/列集不符合契约。"""


class ExecutionError(ProjectError):
    """运行期状态错误（如未 reset 即取帧、非法动作）。"""


class ExternalServiceError(ProjectError):
    """外部服务失败（LLM 网络/鉴权/超时）。"""


def exit_code_for(exc: BaseException) -> int:
    """异常 → CLI 退出码（接口规范 §2.1）。"""
    if isinstance(exc, (ConfigError, ValidationError)):
        return EXIT_CONFIG
    if isinstance(exc, ExternalServiceError):
        return EXIT_EXTERNAL
    if isinstance(exc, ProjectError):
        return EXIT_BIZ
    return EXIT_BIZ
