# -*- coding: utf-8 -*-
"""
引导策略行为树/llm_bt/boundary_constraints.py（步骤 2.6 D2.6-1）
================================================================
第一优先级边界约束守卫：纯 Python 确定性规则，包装/拦截战术执行子树
每一步下发的 Dict 动作（flight/weapon）。规则按固定优先级覆盖或限幅动作，
不是可被 Selector 绕过的条件分支。

约束（阈值默认对齐 common/bvr_combat_env.DEFAULT_CONFIG）：
  1. min_altitude   离地高度 < agl_min_m            → 满油门拉起
  2. min_speed      真空速 < vmin_m * 1.2           → 满油门低头加速
  3. over_g         法向过载超出 [nz_min, nz_max]   → 限制 roll/pitch 步进并放松拉杆
  4. max_speed      Mach > mach_max                 → 收油门、抬头减速
  5. no_fly_zone    进入配置的圆形/多边形禁飞区     → 朝最近脱离方向转弯
  6. fuel_low       燃油比例 < fuel_rtb_frac        → 强制返航（复用指令回路）

设计要点：
- 每个约束带滞回（enter/exit 阈值不同），避免阈值附近抖动；
- 触发时写入结构化日志（step/constraint/value/threshold/action_override）；
- 动作覆盖顺序固定：瞬时飞行安全 > 导航/任务约束；
- 不修改环境物理/奖励/终止逻辑，仅改变本步动作。

用法：
    guard = BoundaryConstraints(env_cfg=env.cfg)
    action = guard.apply(bb.action, bb)
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .executor import (IDX_H, IDX_MACH, IDX_PHI, IDX_PSI, IDX_V,
                       _bank_to_heading, _hold_altitude, _hold_speed)

# 观测索引（与 executor 保持一致，单独列出便于阅读）
OBS_NZ = 8
OBS_FUEL_FRAC = 10

DEFAULT_BOUNDARY_CONFIG = {
    "nz_max_g": 5.5,
    "nz_min_g": -2.5,
    "vmin_mps": 90.0,
    "vmin_margin": 1.2,
    "mach_max": 1.65,
    "agl_min_m": 500.0,
    "fuel_rtb_frac": 0.20,
    # 滞回量
    "agl_exit_margin_m": 150.0,
    "vmin_exit_margin": 0.10,
    "mach_exit_margin": 0.05,
    "nz_exit_margin_g": 0.30,
    # 过载触发时的操纵变化率上限（每决策步）
    "roll_rate_limit": 0.12,
    "pitch_rate_limit": 0.08,
    # 禁飞区：[{"name","type":"circle","center":[east,north],"radius_m":R}
    #          {"name","type":"polygon","vertices":[[e,n],...]}]
    "no_fly_zones": [],
}

# 动作覆盖优先级（数值越小越优先）
PRIORITY = ("min_altitude", "min_speed", "over_g", "max_speed",
            "no_fly_zone", "fuel_low")

OVERRIDE = {
    "min_altitude": "pull_up",
    "min_speed": "accelerate",
    "over_g": "limit_control_rate",
    "max_speed": "decelerate",
    "no_fly_zone": "exit_no_fly_zone",
    "fuel_low": "return_to_base",
}


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _point_in_polygon(point, vertices) -> bool:
    """射线法判断点是否在多边形内（vertices 为 [(e,n), ...]）。"""
    e, n = point
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        ei, ni = vertices[i]
        ej, nj = vertices[j]
        if ((ni > n) != (nj > n)) and \
                (e < (ej - ei) * (n - ni) / (nj - ni + 1e-12) + ei):
            inside = not inside
        j = i
    return inside


class BoundaryConstraints:
    """最高优先级边界约束守卫。"""

    def __init__(self, config: dict | None = None, env_cfg: dict | None = None):
        cfg = dict(DEFAULT_BOUNDARY_CONFIG)
        if config:
            cfg.update(config)
        if env_cfg:
            # 安全阈值以环境配置为唯一来源（显式映射，避免键名漂移）
            cfg["nz_max_g"] = float(env_cfg.get("safety_nz_max_g", cfg["nz_max_g"]))
            cfg["nz_min_g"] = float(env_cfg.get("safety_nz_min_g", cfg["nz_min_g"]))
            cfg["mach_max"] = float(env_cfg.get("safety_mach_max", cfg["mach_max"]))
            cfg["vmin_mps"] = float(env_cfg.get("safety_vmin_mps", cfg["vmin_mps"]))
        self.cfg = cfg
        self.reset()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def reset(self):
        self.prev_action = {"flight": [0.55, 0.0, 0.0, 0.0],
                            "weapon": [0, 0, 0]}
        self.active = {name: False for name in PRIORITY}
        self.counts = {name: 0 for name in PRIORITY}
        self.override_steps = {name: 0 for name in PRIORITY}
        self.log = []
        self.last_override = None

    def _set_active(self, name, enter, exit_, value, threshold, step):
        """带滞回的状态更新；激活时写触发日志。"""
        if self.active[name]:
            if exit_:
                self.active[name] = False
                self.log.append({"step": int(step), "constraint": name,
                                 "event": "clear", "value": float(value),
                                 "threshold": float(threshold),
                                 "action_override": OVERRIDE[name]})
        elif enter:
            self.active[name] = True
            self.counts[name] += 1
            self.log.append({"step": int(step), "constraint": name,
                             "event": "trigger", "value": float(value),
                             "threshold": float(threshold),
                             "action_override": OVERRIDE[name]})

    # ------------------------------------------------------------------
    # 观测读取
    # ------------------------------------------------------------------
    @staticmethod
    def _read(bb, idx: int, default: float = 0.0) -> float:
        try:
            return float(bb.read(idx)) if bb is not None else default
        except Exception:
            return default

    def _agl_m(self, bb) -> float:
        """离地高度：优先 env 状态（含地形），否则按 h_sl 近似（平坦地形）。"""
        if bb is not None and getattr(bb, "env", None) is not None:
            try:
                return float(bb.env.own.state_dict()["h_agl_m"])
            except Exception:
                pass
        return self._read(bb, IDX_H)

    def _own_en(self, bb):
        """本机 ENU 东/北位置（env 帧优先，其次 info 注入，供单测）。"""
        if bb is None:
            return None
        env = getattr(bb, "env", None)
        if env is not None:
            try:
                pos = env.get_viz_frame()["own"]["pos"]
                return float(pos[0]), float(pos[1])
            except Exception:
                pass
        for key in ("own_pos_enu_m", "pos_enu_m"):
            pos = (bb.info or {}).get(key)
            if pos is not None and len(pos) >= 2:
                return float(pos[0]), float(pos[1])
        return None

    # ------------------------------------------------------------------
    # 约束求值
    # ------------------------------------------------------------------
    def _zones(self, bb, pos):
        """返回命中的禁飞区 (zone, value, threshold) 或 None。"""
        zones = self.cfg.get("no_fly_zones") or []
        if pos is None:
            return None
        for zone in zones:
            ztype = str(zone.get("type", "circle")).lower()
            if ztype == "polygon":
                verts = [(float(v[0]), float(v[1])) for v in zone.get("vertices", [])]
                if len(verts) >= 3 and _point_in_polygon(pos, verts):
                    return zone, 0.0, 0.0
            else:
                ce, cn = [float(v) for v in zone.get("center", [0.0, 0.0])]
                radius = float(zone.get("radius_m", 0.0))
                dist = math.hypot(pos[0] - ce, pos[1] - cn)
                if dist < radius:
                    return zone, dist, radius
        return None

    def evaluate(self, bb):
        """更新全部约束状态，返回按优先级排序的激活约束名列表。"""
        step = int(getattr(bb, "step", 0)) if bb is not None else 0
        agl = self._agl_m(bb)
        v = self._read(bb, IDX_V)
        mach = self._read(bb, IDX_MACH)
        nz = self._read(bb, OBS_NZ)
        fuel = self._read(bb, OBS_FUEL_FRAC)
        pos = self._own_en(bb)
        zone = self._zones(bb, pos)

        c = self.cfg
        self._set_active(
            "min_altitude", agl < c["agl_min_m"],
            agl > c["agl_min_m"] + c["agl_exit_margin_m"], agl, c["agl_min_m"], step)
        v_thr = c["vmin_mps"] * c["vmin_margin"]
        self._set_active(
            "min_speed", v < v_thr,
            v > v_thr * (1.0 + c["vmin_exit_margin"]), v, v_thr, step)
        self._set_active(
            "over_g", nz > c["nz_max_g"] or nz < c["nz_min_g"],
            (c["nz_min_g"] + c["nz_exit_margin_g"] < nz <
             c["nz_max_g"] - c["nz_exit_margin_g"]), nz,
            c["nz_max_g"] if nz >= 0 else c["nz_min_g"], step)
        self._set_active(
            "max_speed", mach > c["mach_max"],
            mach < c["mach_max"] - c["mach_exit_margin"], mach, c["mach_max"], step)
        self._set_active(
            "no_fly_zone", zone is not None, zone is None,
            0.0 if zone is None else zone[1],
            0.0 if zone is None else zone[2], step)
        self._set_active(
            "fuel_low", fuel < c["fuel_rtb_frac"], fuel > c["fuel_rtb_frac"] + 0.05,
            fuel, c["fuel_rtb_frac"], step)
        return [n for n in PRIORITY if self.active[n]]

    # ------------------------------------------------------------------
    # 动作覆盖
    # ------------------------------------------------------------------
    def _override(self, name, action, bb):
        fl = action["flight"]
        if name == "min_altitude":
            fl[0] = 1.0
            fl[1] = _clip(0.5 * fl[1], -0.2, 0.2)       # 先改平机翼
            fl[2] = min(float(fl[2]), -0.15)             # 负俯仰 = 抬头
        elif name == "min_speed":
            fl[0] = 1.0
            fl[1] = _clip(0.5 * fl[1], -0.2, 0.2)
            fl[2] = max(float(fl[2]), 0.12)               # 正俯仰 = 低头加速
        elif name == "over_g":
            nz = self._read(bb, OBS_NZ)
            if nz > self.cfg["nz_max_g"]:
                fl[2] = _clip(float(fl[2]) + 0.06, -0.3, 0.3)   # 放松拉杆
            else:
                fl[2] = _clip(float(fl[2]) - 0.06, -0.3, 0.3)   # 减轻负过载
        elif name == "max_speed":
            fl[0] = 0.0
            fl[2] = min(float(fl[2]), -0.04)               # 轻微抬头减速
        elif name == "no_fly_zone":
            self._escape_zone(action, bb)
        elif name == "fuel_low":
            self._return_to_base(action, bb)
        return action

    def _escape_zone(self, action, bb):
        pos = self._own_en(bb)
        if pos is None:
            return
        zones = self.cfg.get("no_fly_zones") or []
        zone = None
        for z in zones:
            if str(z.get("type", "circle")).lower() == "polygon":
                verts = [(float(v[0]), float(v[1])) for v in z.get("vertices", [])]
                if len(verts) >= 3 and _point_in_polygon(pos, verts):
                    zone = z
                    break
            else:
                ce, cn = [float(v) for v in z.get("center", [0.0, 0.0])]
                if math.hypot(pos[0] - ce, pos[1] - cn) < float(z.get("radius_m", 0.0)):
                    zone = z
                    break
        if zone is None:
            return
        if str(zone.get("type", "circle")).lower() == "polygon":
            verts = [(float(v[0]), float(v[1])) for v in zone.get("vertices", [])]
            ce = sum(v[0] for v in verts) / len(verts)
            cn = sum(v[1] for v in verts) / len(verts)
        else:
            ce, cn = [float(v) for v in zone.get("center", [0.0, 0.0])]
        de, dn = pos[0] - ce, pos[1] - cn
        if math.hypot(de, dn) < 1.0:
            # 位于中心：背向当前航向脱离
            heading = math.degrees(self._read(bb, IDX_PSI)) + 180.0
        else:
            heading = math.degrees(math.atan2(de, dn)) % 360.0
        _bank_to_heading(bb, heading)
        _hold_altitude(bb, max(self._read(bb, IDX_H), self.cfg["agl_min_m"] + 1000.0))
        action["flight"][0] = max(float(action["flight"][0]), 0.8)

    def _return_to_base(self, action, bb):
        heading = 0.0
        if bb is not None and getattr(bb, "env", None) is not None:
            pos = self._own_en(bb)
            if pos is not None and math.hypot(pos[0], pos[1]) > 1.0:
                heading = math.degrees(math.atan2(-pos[0], -pos[1])) % 360.0
        _bank_to_heading(bb, heading)
        _hold_altitude(bb, 9000.0)
        _hold_speed(bb, 250.0)
        action["flight"][0] = max(float(action["flight"][0]), 0.75)

    def _limit_rates(self, action):
        """过载约束激活时限制 roll/pitch 每步变化量。"""
        prev = self.prev_action["flight"]
        cur = action["flight"]
        cur[1] = _clip(float(cur[1]), float(prev[1]) - self.cfg["roll_rate_limit"],
                       float(prev[1]) + self.cfg["roll_rate_limit"])
        cur[2] = _clip(float(cur[2]), float(prev[2]) - self.cfg["pitch_rate_limit"],
                       float(prev[2]) + self.cfg["pitch_rate_limit"])

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def apply(self, action: dict, bb):
        """拦截一步动作：先求值约束，再按优先级覆盖/限幅，最后记录。"""
        if action is None:
            action = {"flight": [0.55, 0.0, 0.0, 0.0], "weapon": [0, 0, 0]}
        action.setdefault("flight", [0.55, 0.0, 0.0, 0.0])
        action.setdefault("weapon", [0, 0, 0])
        if bb is not None:
            bb.action = action              # 指令回路 helper 写入同一个 dict
        active = self.evaluate(bb)
        chosen = active[0] if active else None
        self.last_override = OVERRIDE[chosen] if chosen else None
        if chosen:
            self.override_steps[chosen] += 1
            self._override(chosen, action, bb)
        if self.active["over_g"]:
            self._limit_rates(action)
        # 最终包络截断（不改变环境物理，仅保证动作合法）
        fl = action["flight"]
        fl[0] = _clip(float(fl[0]), 0.0, 1.0)
        fl[1] = _clip(float(fl[1]), -1.0, 1.0)
        fl[2] = _clip(float(fl[2]), -1.0, 1.0)
        fl[3] = _clip(float(fl[3]), -1.0, 1.0)
        self.prev_action = copy.deepcopy(action)
        return action

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "boundary_trigger_count": sum(self.counts.values()),
            "boundary_override_steps": sum(self.override_steps.values()),
            "per_constraint": dict(self.counts),
            "per_constraint_override_steps": dict(self.override_steps),
            "last_override": self.last_override,
            "log_entries": len(self.log),
        }
