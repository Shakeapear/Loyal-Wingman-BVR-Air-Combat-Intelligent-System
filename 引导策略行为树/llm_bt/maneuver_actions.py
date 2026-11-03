# -*- coding: utf-8 -*-
"""ManeuverSpec → 可执行控制器（Phase 2）。

每个招式都是 `(bb, params, elapsed) -> py_trees.Status` 的轻量控制器：
- 只写 `bb.action`，不直接推进 JSBSim；
- 复用 executor 的航向/高度/速度回路，因此 raw_fcs/assisted 均由
  `Blackboard.control_mode` 统一解释；
- 长动作返回 RUNNING，由 py_trees 每帧继续 tick；完成/超时返回 SUCCESS/FAILURE。
- 不改变环境物理/奖励/终止逻辑。

这些控制器是“可执行战术骨架”，不是高保真气动模型；后续可用 ACMI/真实
轨迹整定参数。每个招式必须在 `MANEUVER_EXECUTORS` 中登记，行为树只允许
通过 `execute_maneuver(name, parameters)` 调用。
"""
from __future__ import annotations

import math

import numpy as np


def _h():
    # 延迟导入，避免 executor ↔ maneuver_actions 循环。
    from . import executor as ex
    return ex


def _status():
    from .executor import STATUS
    return STATUS


def _elapsed(bb) -> int:
    state = getattr(bb, "_maneuver_state", None) or {}
    return max(0, int(bb.step) - int(state.get("start", bb.step)))


def _start(bb, name: str, params: dict):
    if not isinstance(params, dict):
        raise ValueError("execute_maneuver.parameters 必须为 dict")
    state = getattr(bb, "_maneuver_state", None)
    if not state or state.get("name") != name:
        state = {"name": name, "start": int(bb.step), "params": dict(params)}
        bb._maneuver_state = state
    return state


def _timeout(bb, params: dict, default: float = 30.0) -> bool:
    return _elapsed(bb) >= float(params.get("duration_s", default))


def _speed(bb, value: float):
    _h()._hold_speed(bb, float(value))


def _alt(bb, value: float):
    _h()._hold_altitude(bb, float(value))


def _turn_relative(bb, offset_deg: float, max_bank: float = 45.0):
    psi = math.degrees(bb.read(_h().IDX_PSI))
    _h()._bank_to_heading(bb, psi + float(offset_deg), max_bank_cmd=max_bank)


def _turn_target(bb, offset_deg: float = 0.0, max_bank: float = 45.0):
    psi = math.degrees(bb.read(_h().IDX_PSI))
    if bb.read(_h().IDX_TGT_DETECTED) > 0.5:
        brg = math.degrees(bb.read(_h().IDX_TGT_BRG))
        target = psi + brg + float(offset_deg)
    else:
        target = psi
    _h()._bank_to_heading(bb, target, max_bank_cmd=max_bank)


def _roll_command(bb, bank_deg: float):
    """roll/pitch 直接指令，按控制模式解释。"""
    if getattr(bb, "control_mode", "raw_fcs") == "assisted":
        bb.action["flight"][1] = float(np.clip(bank_deg / 45.0, -1.0, 1.0))
    else:
        bb.action["flight"][1] = float(np.clip(bank_deg / 45.0 * 0.3,
                                              -0.3, 0.3))


def _pitch_command(bb, pitch_norm: float):
    if getattr(bb, "control_mode", "raw_fcs") == "assisted":
        bb.action["flight"][2] = float(np.clip(pitch_norm, -1.0, 1.0))
    else:
        bb.action["flight"][2] = float(np.clip(-pitch_norm * 0.3,
                                              -0.3, 0.3))


# ----------------------------------------------------------------------
# 各招式控制器
# ----------------------------------------------------------------------
def _search(bb, params, elapsed):
    _h()._act_search(bb, params)
    return _status().SUCCESS


def _intercept(bb, params, elapsed):
    if bb.read(_h().IDX_TGT_DETECTED) < 0.5:
        return _status().FAILURE
    _turn_target(bb)
    _alt(bb, bb.read(_h().IDX_H))
    _speed(bb, float(params.get("speed_mps", 280.0)))
    if bb.info.get("in_zone", False):
        return _status().SUCCESS
    return _status().RUNNING


def _lock(bb, params, elapsed):
    return _h()._act_lock(bb, params)


def _crank(bb, params, elapsed):
    if bb.read(_h().IDX_TGT_DETECTED) < 0.5:
        return _status().FAILURE
    if not _h()._own_missiles_in_flight(bb):
        return _status().SUCCESS
    _turn_target(bb, float(params.get("offset_deg", 60.0)))
    _alt(bb, bb.read(_h().IDX_H))
    _speed(bb, float(params.get("speed_mps", 250.0)))
    return _status().RUNNING


def _notch(bb, params, elapsed):
    if bb.read(_h().IDX_TGT_DETECTED) < 0.5:
        return _status().FAILURE
    _turn_target(bb, float(params.get("offset_deg", 90.0)))
    _alt(bb, bb.read(_h().IDX_H) - 100.0)
    _speed(bb, float(params.get("speed_mps", 250.0)))
    return (_status().SUCCESS if _timeout(bb, params, 8.0)
            else _status().RUNNING)


def _break_turn(bb, params, elapsed):
    _turn_relative(bb, float(params.get("heading_offset_deg", 90.0)), 60.0)
    _alt(bb, bb.read(_h().IDX_H))
    _speed(bb, float(params.get("speed_mps", 240.0)))
    return (_status().SUCCESS if _timeout(bb, params, 5.0)
            else _status().RUNNING)


def _barrel_roll(bb, params, elapsed):
    # 三段：建立坡度 → 反向滚转 → 改平；作为可执行骨架，后续按轨迹整定。
    phase = elapsed // 2
    bank = (60.0, -60.0, 0.0)[min(phase, 2)]
    _roll_command(bb, bank)
    _alt(bb, bb.read(_h().IDX_H))
    _speed(bb, float(params.get("speed_mps", 250.0)))
    return (_status().SUCCESS if elapsed >= 6 else _status().RUNNING)


def _defensive_spiral(bb, params, elapsed):
    _roll_command(bb, float(params.get("bank_deg", 60.0)))
    target_alt = max(2500.0, bb.read(_h().IDX_H) - 500.0)
    _alt(bb, target_alt)
    _speed(bb, float(params.get("speed_mps", 250.0)))
    if bb.read(_h().IDX_H) <= 2500.0 or _timeout(bb, params, 10.0):
        return _status().SUCCESS
    return _status().RUNNING


def _launch_and_crank(bb, params, elapsed):
    if not bb.info.get("in_zone", False) or bb.read(_h().IDX_N_MSL) < 0.5:
        return _status().FAILURE
    fired = _h()._act_fire(bb, {})
    if fired != _status().SUCCESS:
        return fired
    return _crank(bb, params, elapsed)


def _extend_and_reengage(bb, params, elapsed):
    # 背向目标脱离，保持能量；结束后由树上层的 lock_target 重新接敌。
    _turn_target(bb, float(params.get("offset_deg", 180.0)))
    _alt(bb, bb.read(_h().IDX_H))
    _speed(bb, float(params.get("speed_mps", 280.0)))
    return (_status().SUCCESS if _timeout(bb, params, 10.0)
            else _status().RUNNING)


def _split_s(bb, params, elapsed):
    _roll_command(bb, 0.0)
    _pitch_command(bb, -0.6)
    _speed(bb, float(params.get("speed_mps", 260.0)))
    return (_status().SUCCESS if elapsed >= 4 else _status().RUNNING)


def _loop(bb, params, elapsed):
    _roll_command(bb, 0.0)
    _pitch_command(bb, 0.7)
    _speed(bb, float(params.get("speed_mps", 280.0)))
    return (_status().SUCCESS if elapsed >= 6 else _status().RUNNING)


def _immelmann(bb, params, elapsed):
    _pitch_command(bb, 0.7)
    _roll_command(bb, 60.0 if elapsed < 3 else 0.0)
    _speed(bb, float(params.get("speed_mps", 280.0)))
    return (_status().SUCCESS if elapsed >= 6 else _status().RUNNING)


def _fuel_rtb(bb, params, elapsed):
    return _h()._act_return_to_base(bb, params)


def _ecm_on(bb, params, elapsed):
    _h()._ecm_ensure_on(bb)
    return _status().SUCCESS


MANEUVER_EXECUTORS = {
    "search_pattern": _search,
    "intercept_closure": _intercept,
    "lock_and_track": _lock,
    "crank": _crank,
    "notch": _notch,
    "break_turn": _break_turn,
    "barrel_roll": _barrel_roll,
    "defensive_spiral": _defensive_spiral,
    "launch_and_crank": _launch_and_crank,
    "extend_and_reengage": _extend_and_reengage,
    "split_s": _split_s,
    "loop": _loop,
    "immelmann": _immelmann,
    "fuel_rtb": _fuel_rtb,
    "ecm_on": _ecm_on,
}


def run_maneuver(bb, name: str, params: dict | None = None):
    params = dict(params or {})
    if name not in MANEUVER_EXECUTORS:
        raise ValueError(
            f"未实现招式：{name}；可用：{sorted(MANEUVER_EXECUTORS)}")
    _start(bb, name, params)
    return MANEUVER_EXECUTORS[name](bb, params, _elapsed(bb))


__all__ = ["MANEUVER_EXECUTORS", "run_maneuver"]
