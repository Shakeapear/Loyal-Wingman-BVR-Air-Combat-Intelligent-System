# -*- coding: utf-8 -*-
"""
common/radar_model.py（交付物 D2.2-2）
======================================
机载火控雷达（FireControlRadar）、雷达告警接收机（RWR）、导弹逼近告警（MAWS）
简化模型（确定性，可被 BVRCombatEnv 逐帧调用）。

模型要点（参数来源见 common/README_战场要素.md §参数来源表）：
- FireControlRadar：
  * 最大探测距离 R_max(σ) = R_max_ref · (σ/σ_ref)^(1/4)
    （雷达方程：R ∝ σ^(1/4)，σ 为目标 RCS m²；
    R_max_ref=80 km @ σ_ref=5 m² 取自项目计划书 2.1 节雷达约束）；
  * 方位/俯仰扫描范围 ±60°（超出体积即不可探测）；
  * 最多同时跟踪 10 个目标（按距离优先）；
  * 目标状态机：TRACK（跟踪）→ LOST（丢失记忆，持续 N 次扫描可重捕获回 TRACK）；
  * 跟踪保持距离取探测距离的 1.1 倍（滞后避免边界抖动）。
- RWR（无源告警）：探测敌机火控雷达辐射；
  * 仅能给出粗略方位（15° 量化，精度约 ±10°），不能测距；
  * 敌雷达对我进入单目标跟踪（STT，锁定）时给出"锁定告警"。
- MAWS（紫外/红外逼近告警近似）：探测来袭导弹；
  * 探测距离 8 km（紫外告警典型 3~10 km 量级，取中值），近似全向；
  * 仅对接近（径向接近率 > 50 m/s）目标告警，估计到达时间 TTA = r / V_c。

所有量均为 SI 单位（m、m/s、m²），角度 API 边界用 deg，内部 rad。

典型用法（BVRCombatEnv 中）：
    from common.radar_model import FireControlRadar, RWR, MAWS

    radar = FireControlRadar()
    targets = [{"id": "t1", "pos": np.array([30000.0, 0.0, 6000.0]), "rcs": 5.0}]
    radar.update(targets, own_pos=np.zeros(3), own_heading_deg=0.0)
    print(radar.state("t1"))     # 'TRACK'
    print(radar.track_range("t1"))   # 当前距离 m
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def wrap_pm180(deg):
    """角度归一化到 [-180, 180) deg。"""
    return float((np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0)


def _rel_frame(pos, own_pos, heading_rad):
    """目标绝对位置 -> 雷达体轴系相对位置（x 轴 = 雷达视轴方向）。"""
    rel = np.asarray(pos, dtype=float) - np.asarray(own_pos, dtype=float)
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot @ rel


class FireControlRadar:
    """机载火控雷达简化模型（搜索/跟踪状态机）。

    属性（类常量）：
        R_MAX_REF   : 参考最大探测距离 80 km（对 σ_ref 目标）
        RCS_REF     : 参考 RCS 5 m²
        AZ_LIMIT    : 方位扫描范围 ±60°（deg）
        EL_LIMIT    : 俯仰扫描范围 ±60°（deg）
        MAX_TRACKS  : 最大同时跟踪目标数 10
        TRACK_RANGE_FACTOR : 跟踪保持距离 = 探测距离 × 1.1
        REACQUIRE_SCANS    : 丢失后连续 N 次更新内重新满足探测条件才重捕获
        P_DETECT     : 单次探测概率（默认 1.0，确定性模型；可设 <1 引入随机）
    """

    R_MAX_REF = 80e3
    RCS_REF = 5.0
    AZ_LIMIT = 60.0
    EL_LIMIT = 60.0
    MAX_TRACKS = 10
    TRACK_RANGE_FACTOR = 1.1
    REACQUIRE_SCANS = 3
    P_DETECT = 1.0

    # 目标状态
    TRACK = "TRACK"    # 稳定跟踪
    LOST = "LOST"      # 丢失（保留记忆，可重捕获）

    def __init__(self, r_max_ref=80e3, rcs_ref=5.0, az_limit=60.0, el_limit=60.0,
                 max_tracks=10, track_range_factor=1.1, reacquire_scans=3,
                 p_detect=1.0, seed=None):
        self.r_max_ref = r_max_ref
        self.rcs_ref = rcs_ref
        self.az_limit = az_limit
        self.el_limit = el_limit
        self.max_tracks = max_tracks
        self.track_range_factor = track_range_factor
        self.reacquire_scans = reacquire_scans
        self.p_detect = p_detect
        self.rng = np.random.default_rng(seed)
        # tid -> {"state", "range", "az", "el", "rcs", "lost_count"}
        self._tracks: dict = {}

    # ---- 雷达方程 ----
    def max_detection_range(self, rcs: float) -> float:
        """R_max(σ) = R_max_ref·(σ/σ_ref)^(1/4)。"""
        return self.r_max_ref * (max(rcs, 1e-6) / self.rcs_ref) ** 0.25

    # ---- 查询接口 ----
    def state(self, tid) -> str | None:
        """目标当前状态：'TRACK' / 'LOST' / None（从未被探测到）。"""
        t = self._tracks.get(tid)
        return t["state"] if t else None

    def track_range(self, tid) -> float | None:
        t = self._tracks.get(tid)
        return t["range"] if t else None

    @property
    def tracked_ids(self):
        return sorted(tid for tid, t in self._tracks.items() if t["state"] == self.TRACK)

    def _angle_of(self, rel):
        """相对位置 -> (距离 m, 方位 deg, 俯仰 deg)。"""
        r = float(np.linalg.norm(rel))
        if r < 1e-9:
            return 0.0, 0.0, 0.0
        az = float(np.degrees(np.arctan2(rel[1], rel[0])))   # 右正左负
        el = float(np.degrees(np.arcsin(np.clip(rel[2] / r, -1.0, 1.0))))
        return r, az, el

    def _detected(self, r, az, el, rcs):
        """当前时刻是否满足探测条件（距离/角度体积/概率）。"""
        if abs(az) > self.az_limit or abs(el) > self.el_limit:
            return False
        if r > self.max_detection_range(rcs):
            return False
        if self.p_detect < 1.0:
            return bool(self.rng.random() < self.p_detect)
        return True

    def update(self, targets, own_pos, own_heading_deg=0.0):
        """以当前时刻的目标列表更新雷达状态机。

        参数：
            targets         : [{"id": str, "pos": np.array(3) ENU, "rcs": float}]
            own_pos         : 本机位置 np.array(3)（ENU，m）
            own_heading_deg : 本机航向 deg（雷达视轴默认与机头一致）
        """
        heading = np.deg2rad(own_heading_deg)
        # 1) 计算各目标的距离/方位/俯仰与探测标记
        detections = {}
        for t in targets:
            rel = _rel_frame(t["pos"], own_pos, heading)
            r, az, el = self._angle_of(rel)
            detections[t["id"]] = {
                "range": r, "az": az, "el": el, "rcs": float(t.get("rcs", self.rcs_ref)),
                "detected": self._detected(r, az, el, float(t.get("rcs", self.rcs_ref))),
            }
        # 2) 状态机更新
        for tid, d in detections.items():
            rec = self._tracks.get(tid)
            if rec is None:
                if d["detected"]:
                    rec = {"state": self.TRACK, "range": d["range"], "az": d["az"],
                           "el": d["el"], "rcs": d["rcs"], "lost_count": 0}
                else:
                    continue
            elif rec["state"] == self.TRACK:
                # 跟踪保持范围放宽（滞后），仍满足角度体积
                keep_r = self.max_detection_range(d["rcs"]) * self.track_range_factor
                if (abs(d["az"]) <= self.az_limit and abs(d["el"]) <= self.el_limit
                        and d["range"] <= keep_r):
                    rec.update({"range": d["range"], "az": d["az"], "el": d["el"], "rcs": d["rcs"]})
                else:
                    rec["state"] = self.LOST
                    rec["lost_count"] = 0
            elif rec["state"] == self.LOST:
                if d["detected"]:
                    rec["lost_count"] += 1
                    if rec["lost_count"] >= self.reacquire_scans:
                        rec.update({"state": self.TRACK, "range": d["range"],
                                    "az": d["az"], "el": d["el"], "rcs": d["rcs"],
                                    "lost_count": 0})
                else:
                    rec["lost_count"] = 0
            self._tracks[tid] = rec
        # 3) 跟踪容量限制：最多 max_tracks 个 TRACK，超限时按距离优先（近者保留）
        tracked = [tid for tid, t in self._tracks.items() if t["state"] == self.TRACK]
        if len(tracked) > self.max_tracks:
            tracked.sort(key=lambda tid: self._tracks[tid]["range"])
            for tid in tracked[self.max_tracks:]:
                self._tracks[tid]["state"] = self.LOST
                self._tracks[tid]["lost_count"] = 0


class RWR:
    """雷达告警接收机（无源）简化模型。

    - 探测敌方机载雷达辐射源（灵敏度范围内全部上报）；
    - 方位仅粗略估计：15° 量化（典型 RWR 精度 ±10° 量级）；
    - 无源测角不能测距：range 恒为 None；
    - 当任一敌方雷达对我进入单目标跟踪（STT 锁定）时给出锁定告警。
    """

    SENSITIVITY_RANGE = 250e3   # 灵敏度距离 m（典型 RWR 对机载火控雷达作用距离）
    BEARING_QUANT_DEG = 15.0    # 方位量化间隔 deg

    def __init__(self, sensitivity_range=250e3, bearing_quant_deg=15.0):
        self.sensitivity_range = sensitivity_range
        self.bearing_quant_deg = bearing_quant_deg

    def update(self, emitters, own_pos, own_heading_deg=0.0):
        """emitters: [{"id", "pos": np.array(3), "mode": 'search'|'lock',
        "lock_target": 目标 id（lock 时给出，'self' 表示锁定本机）}]

        返回 {"alarm": bool, "threats": [{"id", "mode", "bearing_deg", "range": None}]}
        """
        heading = np.deg2rad(own_heading_deg)
        threats = []
        alarm = False
        for em in emitters:
            rel = _rel_frame(em["pos"], own_pos, heading)
            r = float(np.linalg.norm(rel))
            if r > self.sensitivity_range:
                continue
            az = float(np.degrees(np.arctan2(rel[1], rel[0])))
            bearing = round(wrap_pm180(az) / self.bearing_quant_deg) * self.bearing_quant_deg
            mode = em.get("mode", "search")
            threats.append({"id": em["id"], "mode": mode,
                            "bearing_deg": float(wrap_pm180(bearing)),
                            "range": None})   # RWR 无源定位，距离不可测
            if mode == "lock" and em.get("lock_target") == "self":
                alarm = True
        return {"alarm": alarm, "threats": threats}


class MAWS:
    """导弹逼近告警系统（紫外/红外告警简化近似）。

    - 探测来袭导弹（发动机羽烟），探测距离 8 km、近似全向；
    - 仅对径向接近的导弹告警（接近率 > min_closing）；
    - 估计到达时间 TTA = r / V_c（r 为当前距离，V_c 为径向接近率）。
    """

    DETECT_RANGE = 8e3      # 探测距离 m（紫外告警典型 3~10 km 取中值）
    MIN_CLOSING = 50.0      # 最小径向接近率 m/s

    def __init__(self, detect_range=8e3, min_closing=50.0):
        self.detect_range = detect_range
        self.min_closing = min_closing

    def update(self, missiles, own_pos, own_vel):
        """missiles: [{"id", "pos": np.array(3), "vel": np.array(3)}]

        返回 {"alarm": bool, "warnings": [{"id", "range", "closing_rate", "tta"}]}
        """
        warnings = []
        own_pos = np.asarray(own_pos, dtype=float)
        own_vel = np.asarray(own_vel, dtype=float)
        for m in missiles:
            rv = np.asarray(m["pos"], dtype=float) - own_pos
            r = float(np.linalg.norm(rv))
            if r > self.detect_range or r < 1e-9:
                continue
            v_rel = np.asarray(m["vel"], dtype=float) - own_vel
            v_close = -float(np.dot(v_rel, rv)) / r   # >0 表示接近
            if v_close <= self.min_closing:
                continue
            warnings.append({"id": m["id"], "range": r,
                             "closing_rate": v_close, "tta": r / v_close})
        return {"alarm": bool(warnings), "warnings": warnings}
