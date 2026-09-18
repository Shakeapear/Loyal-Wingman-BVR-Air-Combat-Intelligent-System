# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/executor.py（步骤 2.5 组件：仿真执行器）
================================================================
把 LLM 输出的行为树 JSON 解析为可执行的 py_trees 树，并在 BVRCombatEnv
（JSBSim）中以 1 Hz 决策循环执行（计划书步骤 2.5「实现仿真执行器」）。

设计要点：
- 组合节点以**反应式**语义构建（memory=False）：每个决策步从根重新求值，
  相当于 1 Hz 决策策略；Sequence 的条件不满足即失败并回退到下一分支；
- Condition 读取观测（obs 索引与 common/bvr_combat_env.OBS_FIELDS_BVR 对齐）；
- Action 生成 BVRCombatEnv 的 Dict 动作（flight 为指令回路的中间量，
  weapon 为边沿触发）；长动作（navigate/evade/return_to_base）返回 RUNNING
  并持续下发指令，直至完成条件满足；
- 全程记录节点执行日志 tick_counts / log（D2.7 可追溯性要求的雏形）；
- 指令写入语义：动作节点写黑板 action（flight/weapon），反应式组合节点按
  优先级短路求值——每一步只有"获胜分支"的指令应生效；**高优先级分支失败后
  不要让默认巡逻动作覆盖同一步的跟踪/机动指令**（正确树例见 mock_llm：交战
  序列在发射条件不满足时回退到 lock_target 继续跟踪，而非回落到 Default_Patrol）；
- 单机环境限制：maintain_formation 无僚机对象、has_human_command 默认无外部
  输入（黑板上可由调用方置位），二者为接口预留，详见 README。

动作→指令映射复用 2.4 演示策略实测整定的回路增益（F-104 人工通道）：
滚转经坡度内环（防 1 Hz 零阶保持失稳）、升降舵符号相反、油门按速差微调。
"""
from __future__ import annotations

import math
from collections import Counter
from functools import partial

import numpy as np
import py_trees

from .schema import ACTIONS, CONDITIONS, node_name, root_and_children

# ---- 观测索引（对齐 OBS_FIELDS_BVR，共 28 维）----
IDX_H, IDX_V, IDX_MACH, IDX_PHI, IDX_THETA, IDX_PSI = 0, 1, 2, 3, 4, 5
IDX_THROTTLE, IDX_FUEL, IDX_VD, IDX_N_MSL = 9, 10, 11, 12
IDX_TGT_BRG, IDX_TGT_DETECTED = 18, 21
IDX_MAWS = 24

STATUS = py_trees.common.Status
FUEL_LOW_FRAC = 0.15          # is_fuel_low 判定阈值（剩余燃油比例）
FIRE_COOLDOWN_S = 4.0         # 本机连发最小间隔 s（双发齐射战术）
WAYPOINT_ALT_TOL_M = 400.0    # navigate_to_waypoint 到达判定（高度）
WAYPOINT_HDG_TOL_DEG = 15.0   # navigate_to_waypoint 到达判定（航向）
RTB_DIST_TOL_M = 5000.0       # return_to_base 到达判定（距原点）
CRANK_OFFSET_DEG = 60.0       # 发射后 F-pole 偏转角（保持目标在雷达方位边缘）
CRANK_MAX_AGE_S = 30.0        # 导弹在飞判定回退（无 env 时按发射后 30 s）


class Blackboard:
    """行为树黑板：一帧观测 + 环境句柄 + 当步动作 + 执行日志。"""

    def __init__(self, env=None):
        self.env = env
        self.obs = None
        self.info = {}
        self.step = 0
        self.human_command = False
        # 初始指令：巡航油门 + 平飞（保证树失败时也有安全默认指令）
        self.action = {"flight": [0.55, 0.0, 0.0, 0.0], "weapon": [0, 0, 0]}
        self.tick_counts = Counter()
        self.log = []                 # [(step, node_name, status)]
        self._last_fire_step = -99
        self._evade_sign = 1.0
        self._evade_active = False
        self._ecm_requested = False    # ECM 仅边沿开启一次（主防默认开启）

    def read(self, idx) -> float:
        return float(self.obs[idx]) if self.obs is not None else 0.0

    def set_frame(self, obs, info, step):
        self.obs, self.info, self.step = obs, info, int(step)

    def reset_step_weapon(self):
        """清空武器离散动作（边沿触发：仅当步的 1 有效）。"""
        self.action["weapon"] = [0, 0, 0]


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------
# 指令回路（输入意图 → flight 动作中间量）
# ----------------------------------------------------------------------
def _bank_to_heading(bb: Blackboard, heading_deg: float, max_bank_cmd: float = 45.0):
    """航向误差 → 坡度指令 → 副翼（含坡度反馈阻尼；正误差=右转）。"""
    psi_deg = math.degrees(bb.read(IDX_PSI))
    err = _wrap180(heading_deg - psi_deg)
    phi_cmd = float(np.clip(1.2 * err, -max_bank_cmd, max_bank_cmd))
    phi_deg = math.degrees(bb.read(IDX_PHI))
    bb.action["flight"][1] = float(np.clip(0.010 * (phi_cmd - phi_deg), -0.3, 0.3))


def _hold_altitude(bb: Blackboard, alt_m: float, vd_bias_mps: float = 0.0):
    """高度误差 → 垂速指令 → 俯仰（正俯仰=低头，与 F-104 人工通道符号一致）。"""
    vd_cmd = float(np.clip(0.5 * (alt_m - bb.read(IDX_H)) + vd_bias_mps, -25.0, 25.0))
    bb.action["flight"][2] = float(np.clip(-0.01 * (vd_cmd - bb.read(IDX_VD)), -0.3, 0.3))


def _hold_speed(bb: Blackboard, speed_mps: float):
    """速差 → 油门（小增益微调，避免振荡）。"""
    bb.action["flight"][0] = float(np.clip(0.85 + 0.005 * (speed_mps - bb.read(IDX_V)), 0.2, 1.0))


def _ecm_ensure_on(bb: Blackboard):
    """主防默认：ECM 未开启时下发一次开启指令（weapon[2]=1，边沿触发）。

    依据 2.4 演示策略实测：ECM 使敌雷达有效距离 ×0.5，显著推迟敌首发——
    这是长时间对抗存活的关键；故搜索/锁定/规避动作均确保 ECM 开启。
    """
    if bb._ecm_requested or bb.info.get("ecm_on"):
        return
    bb.action["weapon"][2] = 1
    bb._ecm_requested = True


def _own_missiles_in_flight(bb: Blackboard) -> bool:
    """本机导弹是否在飞（优先读环境帧；无 env 时按发射后时间窗回退）。"""
    if bb.env is not None:
        try:
            return len(bb.env.get_viz_frame()["own_missiles"]) > 0
        except Exception:
            pass
    return bb._last_fire_step >= 0 and (bb.step - bb._last_fire_step) <= CRANK_MAX_AGE_S


# ----------------------------------------------------------------------
# 动作节点实现（签名 (bb, args) -> Status）
# ----------------------------------------------------------------------
def _act_navigate(bb: Blackboard, args: dict):
    _bank_to_heading(bb, float(args["heading_deg"]))
    _hold_altitude(bb, float(args["altitude_m"]))
    _hold_speed(bb, float(args["speed_mps"]))
    alt_ok = abs(bb.read(IDX_H) - float(args["altitude_m"])) < WAYPOINT_ALT_TOL_M
    psi_deg = math.degrees(bb.read(IDX_PSI))
    hdg_ok = abs(_wrap180(float(args["heading_deg"]) - psi_deg)) < WAYPOINT_HDG_TOL_DEG
    return STATUS.SUCCESS if (alt_ok and hdg_ok) else STATUS.RUNNING


def _act_search(bb: Blackboard, args: dict):
    """雷达自动扫描（环境内 FireControlRadar 恒开），保持平飞搜索目标；确保 ECM 开启。"""
    _ecm_ensure_on(bb)
    _bank_to_heading(bb, math.degrees(bb.read(IDX_PSI)))
    _hold_altitude(bb, bb.read(IDX_H))
    _hold_speed(bb, 240.0)
    return STATUS.SUCCESS


def _act_lock(bb: Blackboard, args: dict):
    if bb.read(IDX_TGT_DETECTED) < 0.5:
        return STATUS.FAILURE
    _ecm_ensure_on(bb)
    psi_deg = math.degrees(bb.read(IDX_PSI))
    tgt_brg_deg = math.degrees(bb.read(IDX_TGT_BRG))
    if _own_missiles_in_flight(bb):
        # F-pole：导弹在飞期间 crank 偏转 60°（背向目标一侧），保持雷达跟踪并拉开敌弹拦截
        crank_sign = -1.0 if tgt_brg_deg >= 0.0 else 1.0
        _bank_to_heading(bb, psi_deg + tgt_brg_deg + crank_sign * CRANK_OFFSET_DEG)
    else:
        _bank_to_heading(bb, psi_deg + tgt_brg_deg)
    _hold_altitude(bb, bb.read(IDX_H))
    return STATUS.SUCCESS


def _act_fire(bb: Blackboard, args: dict):
    if not bb.info.get("in_zone", False) or bb.read(IDX_N_MSL) < 0.5:
        return STATUS.FAILURE
    if bb.step - bb._last_fire_step < FIRE_COOLDOWN_S:
        return STATUS.FAILURE
    bb.action["weapon"][0] = 1
    bb._last_fire_step = bb.step
    return STATUS.SUCCESS


def _act_evade(bb: Blackboard, args: dict):
    """来袭导弹规避：大坡度偏转 + 下降 + 满油门；TTA<1.5 s 末端急转。"""
    incoming = bb.read(IDX_MAWS) > 0.5 or bool(bb.info.get("maws_alarm", False))
    if not incoming:
        bb._evade_active = False
        return STATUS.SUCCESS
    _ecm_ensure_on(bb)
    if not bb._evade_active:
        bb._evade_active = True
        if bb.read(IDX_TGT_DETECTED) > 0.5:
            # 背向目标偏转（目标在右 → 左转脱离）
            bb._evade_sign = -1.0 if bb.read(IDX_TGT_BRG) >= 0.0 else 1.0
    tta = float(bb.info.get("maws_tta", 99.0))
    bank = 65.0 if tta < 1.5 else 55.0
    phi_cmd = bank * bb._evade_sign
    phi_deg = math.degrees(bb.read(IDX_PHI))
    roll = float(np.clip(0.010 * (phi_cmd - phi_deg), -0.3, 0.3))
    vd_bias = -15.0 if bb.read(IDX_H) > 2500.0 else 0.0
    _hold_altitude(bb, bb.read(IDX_H), vd_bias_mps=vd_bias)
    bb.action["flight"][0] = 1.0
    bb.action["flight"][1] = roll
    return STATUS.RUNNING


def _act_formation(bb: Blackboard, args: dict):
    """编队保持：单机环境无僚机对象，按接口预留处理为平飞保持（见 README 限制）。"""
    _bank_to_heading(bb, math.degrees(bb.read(IDX_PSI)))
    _hold_altitude(bb, bb.read(IDX_H))
    _hold_speed(bb, 240.0)
    return STATUS.SUCCESS


def _act_return_to_base(bb: Blackboard, args: dict):
    if bb.env is None:
        return STATUS.FAILURE
    pos = bb.env.get_viz_frame()["own"]["pos"]
    east, north = float(pos[0]), float(pos[1])
    if math.hypot(east, north) < RTB_DIST_TOL_M:
        return STATUS.SUCCESS
    heading_deg = math.degrees(math.atan2(-east, -north)) % 360.0   # 朝原点
    _bank_to_heading(bb, heading_deg)
    _hold_altitude(bb, 9000.0)
    _hold_speed(bb, 250.0)
    return STATUS.RUNNING


ACT_FUNCS = {
    "navigate_to_waypoint": _act_navigate,
    "search_target": _act_search,
    "lock_target": _act_lock,
    "fire_missile": _act_fire,
    "evade_missile": _act_evade,
    "maintain_formation": _act_formation,
    "return_to_base": _act_return_to_base,
}

COND_FUNCS = {
    "has_target_detected": lambda bb: bb.read(IDX_TGT_DETECTED) > 0.5,
    "is_in_launch_zone": lambda bb: bool(bb.info.get("in_zone", False)),
    "is_missile_incoming": lambda bb: (bb.read(IDX_MAWS) > 0.5
                                       or bool(bb.info.get("maws_alarm", False))),
    "has_weapon_remaining": lambda bb: bb.read(IDX_N_MSL) > 0.5,
    "is_fuel_low": lambda bb: bb.read(IDX_FUEL) < FUEL_LOW_FRAC,
    "has_human_command": lambda bb: bool(bb.human_command),
}


# ----------------------------------------------------------------------
# py_trees 节点与构建
# ----------------------------------------------------------------------
class _LoggedBehaviour(py_trees.behaviour.Behaviour):
    """带黑板日志的同步节点（执行日志供 Tester 覆盖率/可追溯性统计）。"""

    def __init__(self, name, fn, blackboard: Blackboard):
        super().__init__(name)
        self._fn = fn
        self.bb = blackboard

    def update(self):
        status = self._fn(self.bb)
        self.bb.tick_counts[self.name] += 1
        self.bb.log.append((self.bb.step, self.name, status.name))
        return status


def _action_factory(action_name: str, args: dict, bb: Blackboard):
    merged = {**ACTIONS[action_name], **args}
    return partial(ACT_FUNCS[action_name], args=merged)


def build_tree(data: dict, blackboard: Blackboard) -> py_trees.behaviour.Behaviour:
    """行为树 JSON → py_trees 树（根节点）。要求 JSON 已通过 schema 校验。"""
    root_id, children = root_and_children(data)
    if root_id is None:
        raise ValueError("行为树没有根节点，无法构建执行器")
    by_id = {n["id"]: n for n in data["nodes"]}

    def make(nid):
        node = by_id[nid]
        ntype = node["type"]
        if ntype == "Action":
            fn = _action_factory(node["name"], node.get("args") or {}, blackboard)
            return _LoggedBehaviour(node_name(node), fn, blackboard)
        if ntype == "Condition":
            name = node["name"]
            fn = lambda bb, _n=name: (STATUS.SUCCESS if COND_FUNCS[_n](bb)
                                      else STATUS.FAILURE)  # noqa: E731
            return _LoggedBehaviour(name, fn, blackboard)
        kids = [make(c) for c in children.get(nid, [])]
        label = node.get("label") or ntype
        if ntype == "Selector":
            return py_trees.composites.Selector(label, memory=False, children=kids)
        if ntype == "Sequence":
            return py_trees.composites.Sequence(label, memory=False, children=kids)
        if ntype == "Parallel":
            return py_trees.composites.Parallel(
                label, policy=py_trees.common.ParallelPolicy.SuccessOnAll(), children=kids)
        raise ValueError(f"未知节点类型: {ntype}")

    return make(root_id)


def tick_once(root, bb: Blackboard, obs, info, step):
    """执行一个决策步：更新黑板 → tick → 返回本步动作（Dict 动作空间）。"""
    bb.set_frame(obs, info, step)
    bb.reset_step_weapon()
    root.tick_once()
    return bb.action
