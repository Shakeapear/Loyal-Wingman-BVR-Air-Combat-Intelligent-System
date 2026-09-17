# -*- coding: utf-8 -*-
"""
可视化工具/visualization/recorder.py（回放数据层）
================================================================
EpisodeRecorder：逐决策步捕获 BVRCombatEnv.get_viz_frame() 快照，
保存/加载 CSV（每步一行；导弹列表与事件以 JSON 字符串列存储，
与 trajectory_v1 采集规范风格对齐，SI 单位，角度列为 deg）。

CSV 列见 CSV_FIELDS；加载后重建的帧字典与 get_viz_frame() 键集一致，
可直接喂给 offscreen.render_episode / acmi.export_acmi。
录制文件自包含绘制所需全部信息（含攻击区扇形扫描角），
离线回放 / ACMI 转换均不需要智能体库在场。
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

CSV_FIELDS = [
    "steps", "t",
    "own_e_m", "own_n_m", "own_h_m", "own_psi_deg", "own_phi_deg", "own_theta_deg",
    "own_v_mps", "own_mach", "own_nz_g", "own_fuel_lbs", "own_throttle",
    "own_n_left", "own_ecm",
    "enemy_e_m", "enemy_n_m", "enemy_h_m", "enemy_psi_deg", "enemy_phi_deg",
    "enemy_theta_deg", "enemy_v_mps", "enemy_mach", "enemy_n_left",
    "dist_m", "radar_state", "enemy_radar_state",
    "rwr_alarm", "rwr_bearing_deg", "maws_alarm", "maws_tta_s", "in_zone",
    "r_max_own_m", "r_min_own_m", "r_nez_own_m", "r_max_enemy_m", "r_min_enemy_m",
    "radar_az_limit_deg",
    "reward", "terminated_reason", "events_json",
    "own_missiles_json", "enemy_missiles_json",
]


class EpisodeRecorder:
    """episode 逐帧记录器（内存帧列表 + CSV 持久化）。"""

    def __init__(self, meta=None):
        self.frames = []
        # 地理原点（ACMI 导出需要；与环境 reset 的 lat0/lon0 一致）
        self.meta = dict(meta or {"lat0_deg": 30.0, "lon0_deg": 120.0})

    def capture(self, frame):
        """追加一帧（get_viz_frame 快照；深拷贝防止外部引用被后续 step 修改）。"""
        self.frames.append(copy.deepcopy(frame))

    def capture_env(self, env):
        self.capture(env.get_viz_frame())

    def __len__(self):
        return len(self.frames)

    # ------------------------------------------------------------------
    def save_csv(self, path):
        """保存为 CSV（每步一行；None 写为空字符串）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for frm in self.frames:
                w.writerow(_frame_to_row(frm))
        return path

    @staticmethod
    def load_csv(path):
        """加载 CSV，返回帧字典列表（键集与 get_viz_frame() 一致）。"""
        frames = []
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                frames.append(_row_to_frame(row))
        return frames


def _num(x):
    return "" if x is None else f"{float(x):.6g}"


def _frame_to_row(frm):
    own, ene = frm["own"], frm["enemy"]
    return {
        "steps": frm["steps"], "t": _num(frm["t"]),
        "own_e_m": _num(own["pos"][0]), "own_n_m": _num(own["pos"][1]),
        "own_h_m": _num(own["pos"][2]),
        "own_psi_deg": _num(np.rad2deg(own["psi_rad"])),
        "own_phi_deg": _num(np.rad2deg(own.get("phi_rad", 0.0))),
        "own_theta_deg": _num(np.rad2deg(own.get("theta_rad", 0.0))),
        "own_v_mps": _num(own["vtrue_mps"]), "own_mach": _num(own["mach"]),
        "own_nz_g": _num(own.get("nz_g", 0.0)), "own_fuel_lbs": _num(own.get("fuel_lbs", 0.0)),
        "own_throttle": _num(own.get("throttle", 0.0)),
        "own_n_left": own["n_left"], "own_ecm": int(bool(own.get("ecm_on", False))),
        "enemy_e_m": _num(ene["pos"][0]), "enemy_n_m": _num(ene["pos"][1]),
        "enemy_h_m": _num(ene["pos"][2]),
        "enemy_psi_deg": _num(np.rad2deg(ene["psi_rad"])),
        "enemy_phi_deg": _num(np.rad2deg(ene.get("phi_rad", 0.0))),
        "enemy_theta_deg": _num(np.rad2deg(ene.get("theta_rad", 0.0))),
        "enemy_v_mps": _num(ene["vtrue_mps"]), "enemy_mach": _num(ene["mach"]),
        "enemy_n_left": ene["n_left"],
        "dist_m": _num(frm["dist_m"]),
        "radar_state": frm.get("radar_state") or "",
        "enemy_radar_state": frm.get("enemy_radar_state") or "",
        "rwr_alarm": int(bool(frm.get("rwr_alarm", False))),
        "rwr_bearing_deg": _num(np.rad2deg(frm.get("rwr_bearing", 0.0))),
        "maws_alarm": int(bool(frm.get("maws_alarm", False))),
        "maws_tta_s": _num(frm.get("maws_tta", 0.0)),
        "in_zone": int(bool(frm.get("in_zone", False))),
        "r_max_own_m": _num(frm.get("r_max_own")), "r_min_own_m": _num(frm.get("r_min_own")),
        "r_nez_own_m": _num(frm.get("r_nez_own")),
        "r_max_enemy_m": _num(frm.get("r_max_enemy")),
        "r_min_enemy_m": _num(frm.get("r_min_enemy")),
        "radar_az_limit_deg": _num(frm.get("radar_az_limit_deg")),
        "reward": _num(frm.get("reward", 0.0)),
        "terminated_reason": frm.get("terminated_reason") or "",
        "events_json": json.dumps(frm.get("events", {}), ensure_ascii=False),
        "own_missiles_json": json.dumps(frm.get("own_missiles", [])),
        "enemy_missiles_json": json.dumps(frm.get("enemy_missiles", [])),
    }


def _f(row, key, default=0.0):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else default


def _f_none(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def _row_to_frame(row):
    own = {
        "pos": [_f(row, "own_e_m"), _f(row, "own_n_m"), _f(row, "own_h_m")],
        "psi_rad": np.deg2rad(_f(row, "own_psi_deg")),
        "phi_rad": np.deg2rad(_f(row, "own_phi_deg")),
        "theta_rad": np.deg2rad(_f(row, "own_theta_deg")),
        "vtrue_mps": _f(row, "own_v_mps"), "mach": _f(row, "own_mach"),
        "h_sl_m": _f(row, "own_h_m"),
        "nz_g": _f(row, "own_nz_g"), "fuel_lbs": _f(row, "own_fuel_lbs"),
        "throttle": _f(row, "own_throttle"),
        "n_left": int(_f(row, "own_n_left")), "ecm_on": bool(int(_f(row, "own_ecm"))),
    }
    enemy = {
        "pos": [_f(row, "enemy_e_m"), _f(row, "enemy_n_m"), _f(row, "enemy_h_m")],
        "psi_rad": np.deg2rad(_f(row, "enemy_psi_deg")),
        "phi_rad": np.deg2rad(_f(row, "enemy_phi_deg")),
        "theta_rad": np.deg2rad(_f(row, "enemy_theta_deg")),
        "vtrue_mps": _f(row, "enemy_v_mps"), "mach": _f(row, "enemy_mach"),
        "h_sl_m": _f(row, "enemy_h_m"),
        "n_left": int(_f(row, "enemy_n_left")),
    }
    return {
        "steps": int(_f(row, "steps")), "t": _f(row, "t"),
        "own": own, "enemy": enemy,
        "own_missiles": json.loads(row.get("own_missiles_json") or "[]"),
        "enemy_missiles": json.loads(row.get("enemy_missiles_json") or "[]"),
        "dist_m": _f(row, "dist_m"),
        "radar_state": row.get("radar_state") or None,
        "enemy_radar_state": row.get("enemy_radar_state") or None,
        "rwr_alarm": bool(int(_f(row, "rwr_alarm"))),
        "rwr_bearing": np.deg2rad(_f(row, "rwr_bearing_deg")),
        "maws_alarm": bool(int(_f(row, "maws_alarm"))),
        "maws_tta": _f(row, "maws_tta_s"),
        "in_zone": bool(int(_f(row, "in_zone"))),
        "r_max_own": _f_none(row, "r_max_own_m"), "r_min_own": _f_none(row, "r_min_own_m"),
        "r_nez_own": _f_none(row, "r_nez_own_m"),
        "r_max_enemy": _f_none(row, "r_max_enemy_m"),
        "r_min_enemy": _f_none(row, "r_min_enemy_m"),
        "radar_az_limit_deg": _f_none(row, "radar_az_limit_deg"),
        "reward": _f(row, "reward"),
        "terminated_reason": row.get("terminated_reason") or None,
        "events": json.loads(row.get("events_json") or "{}"),
    }
