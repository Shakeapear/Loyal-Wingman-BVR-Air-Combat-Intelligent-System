# -*- coding: utf-8 -*-
"""
run_bt_vs_bt.py（行为树 vs 行为树 1v1 对抗）
================================================
把两棵教师行为树分别放在红/蓝两侧，跑同一批固定 seed 的 1v1 BVR 对抗。

实现方式（不改 common/bvr_combat_env.py 的物理、奖励与终止逻辑）：
- 本机侧：沿用 BVRCombatEnv 原有 raw_fcs 通道 + llm_bt.executor；
- 敌机侧：子类覆写 `_enemy_ai()`，用第二棵行为树产生动作，再经
  `assisted_setpoints()` 映射为敌方 AP 的 roll/alt/speed 指令；
- 敌机观测：由环境状态镜像出同一套 ObsV1（28 维），供敌方行为树使用；
- 敌机雷达/RWR/MAWS、攻击区、发射判定均在子类中按对称口径补齐；
- 为了抵消“本机 raw_fcs / 敌机 AP”的通道差异，默认每对树跑两遍并交换红蓝。

用法（在 引导策略行为树/ 目录下）：
    python -X utf8 run_bt_vs_bt.py --seeds 1-20
    python -X utf8 run_bt_vs_bt.py --seeds 1-10 --scene bvr80-120
    python -X utf8 run_bt_vs_bt.py --seeds 1,2,3 --no-swap

产物：outputs/bt_vs_bt/<timestamp>/result.json + report.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_HERE, _REPO_ROOT / "多任务智能体"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import numpy as np

from common.bvr_combat_env import (BVRCombatEnv, DECISION_DT, OBS_HIGH,
                                    OBS_LOW)
from common.constants import wrap_pi
from common.control_mode import assisted_setpoints
from common.missile_model import launch_envelope
from common.radar_model import MAWS, RWR, FireControlRadar
from llm_bt.boundary_constraints import BoundaryConstraints
from llm_bt.executor import Blackboard, build_tree, tick_once
from llm_bt.maneuvers import ManeuverRegistry
from llm_bt.schema import validate_tree_json
from llm_bt.sim_common import parse_seeds
from llm_bt.tactical_execution_tree import tactical_tree_json
from llm_bt.tester import is_safety_violation

LLM_TREE_PATH = _REPO_ROOT / "docs" / "assets" / "3.5_llm_tree_search" / "bt_best.json"
HAND_TREE_LABEL = "handwritten"
LLM_TREE_LABEL = "deepseek_g1c2"


# ----------------------------------------------------------------------
# 树加载 / 校验
# ----------------------------------------------------------------------
def load_named_tree(name: str) -> tuple[dict, str]:
    if name in ("handwritten", "manual", "hand"):
        return tactical_tree_json(), HAND_TREE_LABEL
    if name in ("llm", "deepseek", "g1c2", "bt_best", "search_g1_c2"):
        path = LLM_TREE_PATH
    else:
        path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(f"行为树不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path.stem


def _validate(tree: dict, label: str) -> dict:
    try:
        maneuver_ids = set(ManeuverRegistry.from_csv().ids())
    except Exception:
        maneuver_ids = None
    errors = validate_tree_json(tree, maneuver_ids=maneuver_ids)
    if errors:
        raise ValueError(f"{label} 行为树校验失败: {errors}")
    return tree


# ----------------------------------------------------------------------
# 敌机侧代理（让第二棵树的 map/return/guard 读取敌方自身状态）
# ----------------------------------------------------------------------
class _EnemyEnvProxy:
    def __init__(self, env: "BTVersusBTEnv"):
        self.env = env
        self.own = env.enemy

    def get_viz_frame(self) -> dict:
        s = self.env.enemy.state_dict()
        return {
            "own": {"pos": [float(v) for v in s["pos_enu_m"]]},
            "own_missiles": [m for m in self.env.enemy_missiles if m.alive],
        }


# ----------------------------------------------------------------------
# 双树环境：本机行为树由外层驱动，敌机行为树在本类内部驱动
# ----------------------------------------------------------------------
class BTVersusBTEnv(BVRCombatEnv):
    """在 BVRCombatEnv 上增加一个由行为树控制的敌机（不改原文件）。"""

    def __init__(self, enemy_tree: dict, config: dict | None = None,
                 enemy_control_mode: str = "assisted", use_guard: bool = True):
        super().__init__(config=config)
        self._enemy_tree_json = enemy_tree
        self._enemy_control_mode = enemy_control_mode
        self._use_guard = bool(use_guard)
        self.enemy_proxy = _EnemyEnvProxy(self)
        self.enemy_bb: Blackboard | None = None
        self.enemy_root = None
        self.enemy_guard: BoundaryConstraints | None = None
        self.enemy_rwr = RWR()
        self.enemy_maws = MAWS()
        self.enemy_ecm_on = False
        self.enemy_obs = None
        self.enemy_info: dict = {}
        self.enemy_logic_errors = 0
        self.r_nez_enemy: float | None = None

    # ---------------- 敌机观测/信息镜像 ----------------
    def _enemy_obs_info(self) -> tuple[np.ndarray, dict]:
        s_e = self.enemy.state_dict()
        s_o = self.own.state_dict()
        own_abs = s_o["pos_enu_m"]
        enemy_abs = self._enemy_abs_pos(s_e)
        rel = own_abs - enemy_abs
        dist = float(np.linalg.norm(rel))
        detected = self.enemy_radar.state("own") == FireControlRadar.TRACK

        enemy_hdg_radar = 90.0 - np.rad2deg(s_e["psi_rad"])
        own_locking = self.radar.state("enemy") == FireControlRadar.TRACK
        rwr_info = self.enemy_rwr.update(
            [{"id": "own_radar", "pos": own_abs,
              "mode": "lock" if own_locking else "search",
              "lock_target": "self" if own_locking else None}],
            enemy_abs, enemy_hdg_radar)
        maws_info = self.enemy_maws.update(
            [{"id": f"om{i}", "pos": m.pos.copy(), "vel": m.vel.copy()}
             for i, m in enumerate(self.own_missiles)],
            enemy_abs, s_e["v_enu_mps"])

        tgt = ((rel[0], rel[1], rel[2], dist,
                wrap_pi(np.arctan2(rel[0], rel[1]) - s_e["psi_rad"]),
                s_o["vtrue_mps"], s_o["psi_rad"])
               if detected else (0.0,) * 7)
        obs = np.array([
            s_e["h_sl_m"], s_e["vtrue_mps"], s_e["mach"],
            s_e["phi_rad"], s_e["theta_rad"], s_e["psi_rad"],
            s_e["alpha_rad"], s_e["beta_rad"], s_e["nz_g"],
            s_e["throttle"], s_e["fuel_frac"], s_e["vd_mps"],
            self.enemy_n_left, 1.0 if self.enemy_ecm_on else 0.0,
            *tgt,
            1.0 if detected else 0.0,
            1.0 if rwr_info.get("alarm") else 0.0,
            float(rwr_info.get("threats", [{}])[0].get("bearing_deg", 0.0) *
                  math.pi / 180.0) if rwr_info.get("threats") else 0.0,
            1.0 if maws_info.get("alarm") else 0.0,
            float(maws_info["warnings"][0]["tta"]) if maws_info.get("warnings") else 0.0,
            1.0, 0.0,
        ], dtype=np.float32)
        obs = np.clip(obs, OBS_LOW, OBS_HIGH)
        in_zone = bool(
            detected and self.r_min_enemy is not None
            and self.r_max_enemy is not None
            and self.r_min_enemy <= dist <= self.r_max_enemy)
        info = {
            "steps": int(self.steps), "dist_m": dist,
            "in_zone": in_zone,
            "maws_alarm": bool(maws_info.get("alarm", False)),
            "maws_tta_s": (float(maws_info["warnings"][0]["tta"])
                           if maws_info.get("warnings") else 0.0),
            "rwr_alarm": bool(rwr_info.get("alarm", False)),
            "rwr_bearing_rad": (np.deg2rad(
                float(rwr_info["threats"][0]["bearing_deg"]))
                if rwr_info.get("threats") else 0.0),
            "ecm_on": bool(self.enemy_ecm_on),
            "r_nez_own_m": self.r_nez_enemy,
            "r_max_enemy_m": self.r_max_enemy,
            "r_min_enemy_m": self.r_min_enemy,
            "enemy_radar_state": self.enemy_radar.state("own"),
        }
        return obs, info

    # ---------------- reset ----------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        s_e = self.enemy.state_dict()
        self.enemy_bb = Blackboard(self.enemy_proxy,
                                   control_mode=self._enemy_control_mode,
                                   alt_ref_m=float(s_e["h_sl_m"]))
        self.enemy_root = build_tree(self._enemy_tree_json, self.enemy_bb)
        self.enemy_guard = (BoundaryConstraints(env_cfg=self.cfg)
                            if self._use_guard else None)
        self.enemy_rwr = RWR()
        self.enemy_maws = MAWS()
        self.enemy_ecm_on = False
        self.enemy_logic_errors = 0

        # 敌方 NEZ（基类 reset 只保存 r_max/r_min，这里补 r_nez）
        if not self.cfg.get("compute_envelope", True):
            self.r_nez_enemy = 20e3
        else:
            s_o = self.own.state_dict()
            rel = self._enemy_abs_pos(s_e) - s_o["pos_enu_m"]
            bearing_enemy_deg = float(np.rad2deg(
                wrap_pi(np.arctan2(-rel[0], -rel[1]) - s_e["psi_rad"])))
            try:
                env_enemy = launch_envelope(
                    launcher_h=s_e["h_sl_m"], launcher_speed=s_e["vtrue_mps"],
                    target_h=s_o["h_sl_m"], target_speed=s_o["vtrue_mps"],
                    bearing_deg=bearing_enemy_deg)
                self.r_nez_enemy = float(env_enemy.r_nez)
            except Exception:
                self.r_nez_enemy = 20e3

        self.enemy_obs, self.enemy_info = self._enemy_obs_info()
        return obs, info

    # ---------------- 对称终止：敌机燃油/逃逸 ----------------
    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if not term:
            s_e = self.enemy.state_dict()
            reason = None
            if s_e["fuel_lbs"] < self.cfg["fuel_min_lbs"]:
                reason = "enemy_fuel_out"
            elif float(np.hypot(s_e["pos_enu_m"][0],
                                s_e["pos_enu_m"][1])) > self.cfg["escape_radius_m"]:
                reason = "enemy_escape"
            if reason is not None:
                term, trunc = True, False
                info["terminated_reason"] = reason
                self.last_info["terminated_reason"] = reason
        return obs, reward, term, trunc, info

    # ---------------- 敌机发射 ----------------
    def _enemy_fire_bt(self, dist: float) -> bool:
        if self.enemy_n_left <= 0:
            return False
        if self.enemy_radar.state("own") != FireControlRadar.TRACK:
            return False
        if not (self.r_min_enemy is not None and self.r_max_enemy is not None
                and self.r_min_enemy <= dist <= 0.95 * self.r_max_enemy):
            return False
        s_e = self.enemy.state_dict()
        s_o = self.own.state_dict()
        self._launch_missile(self.enemy_missiles, self._enemy_abs_pos(s_e),
                             s_e["v_enu_mps"], s_o["pos_enu_m"],
                             s_o["v_enu_mps"])
        self.enemy_n_left -= 1
        self._enemy_fired = True
        return True

    # ---------------- 覆写：敌机由行为树驱动 ----------------
    def _enemy_ai(self):
        if self.enemy_root is None or self.enemy_bb is None:
            return
        s_e = self.enemy.state_dict()
        s_o = self.own.state_dict()
        enemy_abs = self._enemy_abs_pos(s_e)
        rel = s_o["pos_enu_m"] - enemy_abs
        dist = float(np.hypot(rel[0], rel[1]))

        action = None
        try:
            action = tick_once(self.enemy_root, self.enemy_bb,
                               self.enemy_obs, self.enemy_info, self.steps,
                               boundary=self.enemy_guard)
        except Exception:
            self.enemy_logic_errors += 1
        if action is None:
            action = {"flight": [0.55, 0.0, 0.0, 0.0], "weapon": [0, 0, 0]}

        flight = np.asarray(action.get("flight", [0.55, 0, 0, 0]), dtype=float)
        if flight.shape != (4,) or not np.all(np.isfinite(flight)):
            self.enemy_logic_errors += 1
            flight = np.array([0.55, 0.0, 0.0, 0.0])
        flight[0] = float(np.clip(flight[0], 0.0, 1.0))
        flight[1:] = np.clip(flight[1:], -1.0, 1.0)
        weapon = np.clip(np.asarray(action.get("weapon", [0, 0, 0]),
                                     dtype=int), 0, 1)

        if int(weapon[2]) == 1:
            self.enemy_ecm_on = not self.enemy_ecm_on
        # 敌机 ECM 对称作用于本机雷达（基类只在本机侧处理 ECM）
        self.radar.r_max_ref = FireControlRadar.R_MAX_REF * (
            self.cfg["ecm_radar_range_factor"] if self.enemy_ecm_on else 1.0)

        setpoints = assisted_setpoints(
            flight, s_e["h_sl_m"], self.cfg, alt_ref_m=self.enemy_bb.alt_ref_m)
        self._enemy_roll_cmd = float(setpoints["roll_rad"])
        self._enemy_alt_cmd_ft = float(setpoints["alt_ft"])
        self.cfg["enemy_speed_fps"] = float(setpoints["speed_fps"])

        self._enemy_fired = False
        if int(weapon[0]) == 1:
            self._enemy_fire_bt(dist)


# ----------------------------------------------------------------------
# 单局 / 批量
# ----------------------------------------------------------------------
def _launch_geometry(env, frame: dict, side: str, step: int,
                     closure_mps: float | None,
                     target_turn_rate_deg_s: float | None,
                     shooter_turn_rate_deg_s: float | None,
                     last_own_fire_step: int | None,
                     last_enemy_fire_step: int | None) -> dict:
    """记录一次发射时刻的几何/能量/对抗状态（用于事后归因）。"""
    if side == "own":
        shooter, target = frame["own"], frame["enemy"]
        r_nez = frame.get("r_nez_own_m")
        r_max = frame.get("r_max_own_m")
        r_min = frame.get("r_min_own_m")
        radar = frame.get("radar_state")
        tree_id = getattr(env, "_own_bt_id", "")
        shooter_maws = bool(frame.get("maws_alarm", False))
        shooter_tta = float(frame.get("maws_tta_s", 0.0))
        enemy_info = getattr(env, "enemy_info", {}) or {}
        target_maws = bool(enemy_info.get("maws_alarm", False))
        target_tta = float(enemy_info.get("maws_tta_s", 0.0))
    else:
        shooter, target = frame["enemy"], frame["own"]
        r_nez = getattr(env, "r_nez_enemy", None)
        r_max = frame.get("r_max_enemy_m")
        r_min = frame.get("r_min_enemy_m")
        radar = frame.get("enemy_radar_state")
        tree_id = getattr(env, "_enemy_bt_id", "")
        shooter_maws = bool(getattr(env, "enemy_info", {}).get("maws_alarm", False))
        shooter_tta = float(getattr(env, "enemy_info", {}).get("maws_tta_s", 0.0))
        target_maws = bool(frame.get("maws_alarm", False))
        target_tta = float(frame.get("maws_tta_s", 0.0))
    los = np.asarray(target["pos"], dtype=float) - np.asarray(shooter["pos"],
                                                              dtype=float)
    dist = float(np.linalg.norm(los))
    back = -los[:2] / max(1e-6, float(np.linalg.norm(los[:2])))
    psi_t = float(target["psi_rad"])
    vt = np.array([math.sin(psi_t), math.cos(psi_t)])
    aspect = math.degrees(math.acos(float(np.clip(vt @ back, -1.0, 1.0))))

    def _since(last):
        return (None if last is None
                else (int(step) - int(last)) * DECISION_DT)

    return {
        "step": int(step), "t": float(step) * DECISION_DT, "side": side,
        "tree_id": tree_id, "range_m": dist,
        "r_nez_m": r_nez, "r_max_m": r_max, "r_min_m": r_min,
        "in_nez": bool(r_nez is not None and dist <= float(r_nez)),
        "in_zone": bool(radar == FireControlRadar.TRACK
                        and r_min is not None and r_max is not None
                        and float(r_min) <= dist <= float(r_max)),
        "radar_track": bool(radar == FireControlRadar.TRACK),
        "aspect_deg": aspect, "closure_mps": closure_mps,
        "target_turn_rate_deg_s": target_turn_rate_deg_s,
        "shooter_turn_rate_deg_s": shooter_turn_rate_deg_s,
        "target_maws": target_maws,
        "target_maws_tta_s": target_tta,
        "shooter_maws": shooter_maws,
        "shooter_maws_tta_s": shooter_tta,
        "own_missiles_in_flight": len(frame.get("own_missiles", [])),
        "enemy_missiles_in_flight": len(frame.get("enemy_missiles", [])),
        "own_n_left_after": int(env.own_n_left),
        "enemy_n_left_after": int(env.enemy_n_left),
        "own_fired_total": int(env.cfg.get("n_missiles", 2) - env.own_n_left),
        "enemy_fired_total": int(env.cfg.get("n_missiles", 2) - env.enemy_n_left),
        "time_since_last_own_fire_s": _since(last_own_fire_step),
        "time_since_last_enemy_fire_s": _since(last_enemy_fire_step),
        "shooter_speed": float(shooter["vtrue_mps"]),
        "target_speed": float(target["vtrue_mps"]),
        "shooter_alt_m": float(shooter["h_sl_m"]),
        "target_alt_m": float(target["h_sl_m"]),
        "outcome": "unknown",       # hit / miss / in_flight
    }


def run_episode(tree_own: dict, tree_enemy: dict, seed: int, max_steps: int,
                scene_cfg: dict, compute_envelope: bool = True,
                own_mode: str = "raw_fcs",
                record_frames: bool = False,
                record_launches: bool = False) -> dict:
    cfg = {"compute_envelope": bool(compute_envelope),
           "max_steps": int(max_steps), "control_mode": own_mode}
    cfg.update(scene_cfg)
    env = BTVersusBTEnv(tree_enemy, config=cfg)
    env._own_bt_id = str(tree_own.get("bt_id", ""))
    env._enemy_bt_id = str(tree_enemy.get("bt_id", ""))

    obs, info = env.reset(seed=int(seed))
    bb = Blackboard(env, control_mode=env.control_mode,
                    alt_ref_m=env._assisted_alt_ref_m)
    root = build_tree(tree_own, bb)
    guard = BoundaryConstraints(env_cfg=env.cfg)
    frames = [env.get_viz_frame()] if record_frames else []
    launches = []
    own_pending, enemy_pending = [], []
    prev_own_psi = prev_enemy_psi = None
    last_own_launch_step = last_enemy_launch_step = None
    last_pre_dist = None
    logic = safety_own = safety_enemy = 0
    own_fired = enemy_fired = 0
    min_dist = float(info.get("dist_m", 1e9))
    reason, steps, t0 = None, 0, time.time()
    try:
        for step in range(int(max_steps)):
            pre_frame = env.get_viz_frame() if record_launches else None
            own_before = env.own_n_left
            enemy_before = env.enemy_n_left
            own_alive_before = len(env.own_missiles) if record_launches else 0
            enemy_alive_before = len(env.enemy_missiles) if record_launches else 0
            pre_dist = float(pre_frame["dist_m"]) if pre_frame else None
            closure = (None if (last_pre_dist is None or pre_dist is None)
                       else last_pre_dist - pre_dist)
            if pre_dist is not None:
                last_pre_dist = pre_dist
            own_psi = float(pre_frame["own"]["psi_rad"]) if pre_frame else None
            enemy_psi = float(pre_frame["enemy"]["psi_rad"]) if pre_frame else None
            own_turn = (None if (prev_own_psi is None or own_psi is None)
                        else math.degrees(wrap_pi(own_psi - prev_own_psi))
                        / DECISION_DT)
            enemy_turn = (None if (prev_enemy_psi is None or enemy_psi is None)
                          else math.degrees(wrap_pi(enemy_psi - prev_enemy_psi))
                          / DECISION_DT)
            action = tick_once(root, bb, obs, info, step, boundary=guard)
            flight = np.asarray(action["flight"], dtype=float)
            if flight.shape != (4,) or not np.all(np.isfinite(flight)):
                logic += 1
                flight = np.array([0.55, 0.0, 0.0, 0.0])
            flight[0] = float(np.clip(flight[0], 0.0, 1.0))
            flight[1:] = np.clip(flight[1:], -1.0, 1.0)
            weapon = np.clip(np.asarray(action["weapon"], dtype=int), 0, 1)

            obs, _reward, term, trunc, info = env.step(
                {"flight": flight.astype(np.float32), "weapon": weapon})
            if record_launches:
                own_new = int(env.own_n_left < own_before)
                enemy_new = int(env.enemy_n_left < enemy_before)
                own_dead = max(0, own_alive_before + own_new
                               - len(env.own_missiles))
                enemy_dead = max(0, enemy_alive_before + enemy_new
                                 - len(env.enemy_missiles))
                hit_enemy = bool(env.last_events.get("hit_enemy", False))
                hit_own = bool(env.last_events.get("hit_own", False))
                for k in range(own_dead):
                    if own_pending:
                        idx = own_pending.pop(0)
                        launches[idx]["outcome"] = (
                            "hit" if (hit_enemy and k == 0) else "miss")
                for k in range(enemy_dead):
                    if enemy_pending:
                        idx = enemy_pending.pop(0)
                        launches[idx]["outcome"] = (
                            "hit" if (hit_own and k == 0) else "miss")
                if own_new:
                    launches.append(_launch_geometry(
                        env, pre_frame, "own", step, closure,
                        enemy_turn, own_turn,
                        last_own_launch_step, last_enemy_launch_step))
                    own_pending.append(len(launches) - 1)
                    last_own_launch_step = step
                if enemy_new:
                    launches.append(_launch_geometry(
                        env, pre_frame, "enemy", step, closure,
                        own_turn, enemy_turn,
                        last_own_launch_step, last_enemy_launch_step))
                    enemy_pending.append(len(launches) - 1)
                    last_enemy_launch_step = step
                prev_own_psi, prev_enemy_psi = own_psi, enemy_psi
            if record_frames:
                frames.append(env.get_viz_frame())
            steps = step + 1
            own_fired = int(info.get("own_fired", own_fired))
            enemy_fired = int(info.get("enemy_fired", enemy_fired))
            min_dist = min(min_dist, float(info.get("dist_m", min_dist)))
            if is_safety_violation(obs, env.cfg):
                safety_own += 1
            env.enemy_obs, env.enemy_info = env._enemy_obs_info()
            if is_safety_violation(env.enemy_obs, env.cfg):
                safety_enemy += 1
            if term or trunc:
                reason = info.get("terminated_reason")
                break
    finally:
        env.close()

    if reason == "enemy_hit":
        winner_side = "own"
    elif reason in ("own_hit", "own_crash", "fuel_out", "escape"):
        winner_side = "enemy"
    elif reason in ("enemy_crash", "enemy_escape", "enemy_fuel_out"):
        winner_side = "own"
    else:
        winner_side = "draw"

    for idx in own_pending + enemy_pending:
        if launches[idx].get("outcome", "unknown") == "unknown":
            launches[idx]["outcome"] = "in_flight"

    return {
        "seed": int(seed), "steps": steps, "terminated_reason": reason,
        "winner_side": winner_side,
        "own_fired": own_fired, "enemy_fired": enemy_fired,
        "own_hit": reason == "enemy_hit",
        "enemy_hit": reason == "own_hit",
        "own_crash": reason == "own_crash",
        "enemy_crash": reason == "enemy_crash",
        "min_dist_m": round(min_dist, 1),
        "safety_violations_own": safety_own,
        "safety_violations_enemy": safety_enemy,
        "logic_errors_own": logic,
        "logic_errors_enemy": env.enemy_logic_errors,
        "wall_s": round(time.time() - t0, 2),
        "own_tick_counts": dict(bb.tick_counts),
        "enemy_tick_counts": dict(env.enemy_bb.tick_counts) if env.enemy_bb else {},
        "frames": frames, "launches": launches,
    }


def _tree_win_record(row: dict, tree: str) -> str:
    if row["winner_side"] == "draw":
        return "draw"
    return "win" if row["winner_side"] == ("own" if tree == "own" else "enemy") else "loss"


def _aggregate(rows: list[dict], label_a: str, label_b: str) -> dict:
    per_tree = {}
    for key in ("a", "b"):
        label = label_a if key == "a" else label_b
        wins = losses = draws = 0
        hits = survivals = crashes = 0
        combat_wins = non_combat_wins = timeouts = 0
        fired = safety = logic = 0
        steps_sum = 0
        for r in rows:
            own_is = r["own_tree"] == label
            side = "own" if own_is else "enemy"
            outcome = _tree_win_record(r, side)
            reason = r.get("terminated_reason")
            wins += outcome == "win"
            losses += outcome == "loss"
            draws += outcome == "draw"
            if outcome == "win":
                if reason == ("enemy_hit" if own_is else "own_hit"):
                    combat_wins += 1
                else:
                    non_combat_wins += 1
            timeouts += reason == "timeout"
            hits += int(r["own_hit"] if own_is else r["enemy_hit"])
            survivals += int(not (r["enemy_hit"] if own_is else r["own_hit"])
                             and not (r["own_crash"] if own_is else r["enemy_crash"]))
            crashes += int(r["own_crash"] if own_is else r["enemy_crash"])
            fired += r["own_fired"] if own_is else r["enemy_fired"]
            safety += (r["safety_violations_own"] if own_is
                       else r["safety_violations_enemy"])
            logic += (r["logic_errors_own"] if own_is
                      else r["logic_errors_enemy"])
            steps_sum += r["steps"]
        n = max(1, wins + losses + draws)
        per_tree[label] = {
            "episodes": n, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": round(wins / n, 4),
            "loss_rate": round(losses / n, 4),
            "draw_rate": round(draws / n, 4),
            "combat_wins": combat_wins,
            "non_combat_wins": non_combat_wins,
            "combat_win_rate": round(combat_wins / n, 4),
            "timeout_episodes": timeouts,
            "timeout_rate": round(timeouts / n, 4),
            "hit_episodes": hits, "hit_rate": round(hits / n, 4),
            "pk_per_launch": round(hits / max(1, fired), 4),
            "survival_episodes": survivals,
            "survival_rate": round(survivals / n, 4),
            "crash_episodes": crashes,
            "missiles_fired": fired,
            "safety_violations": safety,
            "logic_errors": logic,
            "avg_steps": round(steps_sum / n, 1),
        }
    return {
        "trees": {"a": label_a, "b": label_b},
        "episodes": len(rows),
        "head_to_head": {
            label_a: {
                "wins": per_tree[label_a]["wins"],
                "losses": per_tree[label_a]["losses"],
                "draws": per_tree[label_a]["draws"],
            },
            label_b: {
                "wins": per_tree[label_b]["wins"],
                "losses": per_tree[label_b]["losses"],
                "draws": per_tree[label_b]["draws"],
            },
        },
        "per_tree": per_tree,
        "termination_counts": _count(rows, "terminated_reason"),
        "avg_steps": round(sum(r["steps"] for r in rows) / max(1, len(rows)), 1),
        "avg_min_dist_m": round(sum(r["min_dist_m"] for r in rows) / max(1, len(rows)), 1),
        "seed_list": sorted({r["seed"] for r in rows}),
    }


def _count(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out


def _write_report(out_dir: Path, result: dict) -> None:
    agg = result["aggregate"]
    trees = agg["trees"]
    a, b = trees["a"], trees["b"]
    pa, pb = agg["per_tree"][a], agg["per_tree"][b]
    lines = [
        "# 行为树 1v1 对抗结果", "",
        f"- 场景：`{result['scene']}`，max_steps={result['max_steps']}，"
        f"seeds={result['seeds']}",
        f"- 赛制：{result['series_desc']}（交换红蓝以抵消控制通道差异）",
        f"- 对局数：{agg['episodes']}，平均步数 {agg['avg_steps']}，"
        f"平均最近距 {agg['avg_min_dist_m'] / 1000:.1f} km", "",
        "| 树 | 胜 | 负 | 平 | 胜率 | 击落胜 | 非击落胜 | 平局率 | Pk/弹 | 命中局 | 存活局 | 坠毁 | 发射 | 安全违规 | 逻辑错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, p in ((a, pa), (b, pb)):
        lines.append(
            f"| {label} | {p['wins']} | {p['losses']} | {p['draws']} | "
            f"{p['win_rate']:.0%} | {p['combat_wins']} | {p['non_combat_wins']} | "
            f"{p['draw_rate']:.0%} | {p['pk_per_launch']:.1%} | "
            f"{p['hit_episodes']}/{p['episodes']} | "
            f"{p['survival_episodes']}/{p['episodes']} | {p['crash_episodes']} | "
            f"{p['missiles_fired']} | {p['safety_violations']} | {p['logic_errors']} |")
    lines += [
        "", f"- 终止原因分布：`{agg['termination_counts']}`", "",
        "## 逐局结果", "",
        "| seed | 本机树 | 敌机树 | 结局 | 胜方 | 步数 | 最近距 km | 本机发射 | 敌机发射 |",
        "|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in result["episodes"]:
        lines.append(
            f"| {r['seed']} | {r['own_tree']} | {r['enemy_tree']} | "
            f"{r['terminated_reason'] or 'none'} | {r['winner_side']} | "
            f"{r['steps']} | {r['min_dist_m'] / 1000:.1f} | "
            f"{r['own_fired']} | {r['enemy_fired']} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="两棵行为树 1v1 对抗")
    ap.add_argument("--tree-a", default="handwritten",
                    help="树 A（默认 handwritten）")
    ap.add_argument("--tree-b", default="llm",
                    help="树 B（默认 llm = docs/assets/3.5_llm_tree_search/bt_best.json）")
    ap.add_argument("--seeds", default="1-20", help="种子，如 1-20 或 1,2,3")
    ap.add_argument("--scene", default="default", choices=("default", "bvr80-120"))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 时按场景默认（default 240 / bvr80-120 600）")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fast", action="store_true", help="跳过攻击区解算（冒烟）")
    ap.add_argument("--no-swap", action="store_true", help="只跑 A 本机 / B 敌机")
    ap.add_argument("--own-mode", default="raw_fcs",
                    choices=("raw_fcs", "assisted"),
                    help="本机控制通道；assisted 时与敌机 AP 通道对称")
    ap.add_argument("--record-launches", action="store_true",
                    help="记录逐发几何/防御状态/命中结果到 launches.json")
    args = ap.parse_args(argv)

    scene_cfg = ({"scene_dist_min_m": 80000.0, "scene_dist_max_m": 120000.0}
                 if args.scene == "bvr80-120" else {})
    max_steps = int(args.max_steps or (600 if args.scene == "bvr80-120" else 240))
    seeds = parse_seeds(args.seeds)

    try:
        tree_a, label_a = load_named_tree(args.tree_a)
        tree_b, label_b = load_named_tree(args.tree_b)
        _validate(tree_a, label_a)
        _validate(tree_b, label_b)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _HERE / "outputs" / "bt_vs_bt" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = [(label_a, tree_a, label_b, tree_b)]
    if not args.no_swap:
        orders.append((label_b, tree_b, label_a, tree_a))

    rows = []
    all_launches = []
    t0 = time.time()
    print(f"[对抗] {label_a} vs {label_b}  scene={args.scene}  "
          f"max_steps={max_steps}  seeds={seeds}  交换红蓝={not args.no_swap}")
    for own_label, own_tree, enemy_label, enemy_tree in orders:
        for seed in seeds:
            row = run_episode(own_tree, enemy_tree, seed, max_steps, scene_cfg,
                              compute_envelope=not args.fast,
                              own_mode=args.own_mode,
                              record_launches=args.record_launches)
            row["own_tree"] = own_label
            row["enemy_tree"] = enemy_label
            row["winner_tree"] = (
                "draw" if row["winner_side"] == "draw"
                else (own_label if row["winner_side"] == "own" else enemy_label))
            if args.record_launches:
                for launch in row.get("launches", []):
                    launch.update({
                        "seed": int(seed), "own_tree": own_label,
                        "enemy_tree": enemy_label,
                        "winner_side": row["winner_side"],
                        "terminated_reason": row["terminated_reason"],
                    })
                    all_launches.append(launch)
            row.pop("frames", None)
            row.pop("launches", None)
            rows.append(row)
            print(f"[{own_label:>16} vs {enemy_label:<16} seed {seed:>2}] "
                  f"{row['winner_tree']:<16} reason={row['terminated_reason'] or 'none':<10} "
                  f"steps={row['steps']:<3d} min={row['min_dist_m'] / 1000:5.1f}km "
                  f"fired={row['own_fired']}/{row['enemy_fired']}", flush=True)

    agg = _aggregate(rows, label_a, label_b)
    result = {
        "scene": args.scene, "max_steps": max_steps, "seeds": seeds,
        "own_control_mode": args.own_mode,
        "series_desc": (f"{label_a} 本机({args.own_mode}) vs {label_b} 敌机(AP)"
                        + ("" if args.no_swap else f"，并交换为 {label_b} 本机 vs {label_a} 敌机")),
        "tree_sources": {
            label_a: str(_HERE / "llm_bt" / "tactical_execution_tree.py::tactical_tree_json")
            if args.tree_a in ("handwritten", "manual", "hand") else args.tree_a,
            label_b: str(LLM_TREE_PATH) if args.tree_b in ("llm", "deepseek", "g1c2")
            else args.tree_b,
        },
        "aggregate": agg, "episodes": rows,
        "wall_s": round(time.time() - t0, 1),
    }
    if args.record_launches:
        launches_path = out_dir / "launches.json"
        launches_path.write_text(
            json.dumps(all_launches, ensure_ascii=False, indent=2),
            encoding="utf-8")
        result["launches_path"] = str(launches_path)
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(out_dir, result)

    pa = agg["per_tree"][label_a]
    pb = agg["per_tree"][label_b]
    print("-" * 78)
    print(f"[总结果] {label_a}: {pa['wins']}胜{pa['losses']}负{pa['draws']}平 "
          f"(胜率 {pa['win_rate']:.0%}, 击落胜 {pa['combat_wins']}, "
          f"Pk/弹 {pa['pk_per_launch']:.1%}, 平局 {pa['draw_rate']:.0%})")
    print(f"         {label_b}: {pb['wins']}胜{pb['losses']}负{pb['draws']}平 "
          f"(胜率 {pb['win_rate']:.0%}, 击落胜 {pb['combat_wins']}, "
          f"Pk/弹 {pb['pk_per_launch']:.1%}, 平局 {pb['draw_rate']:.0%})")
    print(f"[终止] {agg['termination_counts']}")
    print(f"[输出] {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
