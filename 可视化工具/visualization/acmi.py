# -*- coding: utf-8 -*-
"""
可视化工具/visualization/acmi.py（TacView ACMI 2.2 回放导出）
================================================================
把 get_viz_frame() 帧序列导出为 ACMI 2.2 文本（TacView 公开格式，
规范见 https://www.tacview.net/documentation/acmi/en/ ），
可用 TacView 免费版做 3D 空战回放。本模块为自研实现，仅写文本，
不依赖任何第三方代码（区别于 GPL 的 BVRGym TacviewLogger，仅借鉴思路）。

坐标换算：帧内 ENU（相对 reset 地理原点 lat0/lon0，与环境一致）→
经纬度（1° 纬 = 111320 m，与 jsbsim_bridge 相同的近似）。

已实现的专业级增强（对照 ACMI 2.2 公开规范）：
- 红蓝阵营：首帧 Coalition=Allies/Enemies；
- 雷达锁定连线：radar_state=="TRACK" → LockedTarget=<敌机 id>（失锁时
  以 LockedTarget= 空值移除，并附 LockedTargetMode=0/1 与 LockedTargetRange）；
- 爆炸效果：命中帧被命中方 Health=0（保留 Event=Destroyed）；
- 射击日志：导弹消失时输出 Event=Timeout|SourceId:..|AmmoType:AAM|
  AmmoCount:1|TargetId:..|Outcome:Kill/Miss（命中帧消失的弹→Kill，其余
  →Miss；环境中命中与导弹移出列表发生在同一决策步，见
  bvr_combat_env._advance_own_missiles，故按同帧 events 判定）；
  末帧仍在飞的导弹一并冲刷移除并记 Miss，保证射击日志完整；
- 首帧 Bookmark 标注初始距离；Title 由调用方传入（demo 带 seed）。

用法：
    from visualization.acmi import export_acmi
    export_acmi(frames, "episode.acmi")
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

OWN_ID = 0x100
ENEMY_ID = 0x200
_MSL_MATCH_M = 4000.0   # 跨帧导弹关联阈值（导弹 1 s 约飞 600~900 m，取 4 km 裕度）


def _enu_to_lonlat(e_m, n_m, lat0_deg, lon0_deg):
    lat = lat0_deg + n_m / 111320.0
    lon = lon0_deg + e_m / (111320.0 * math.cos(math.radians(lat0_deg)))
    return lon, lat


def _deg(rad):
    return math.degrees(rad)


class _MissileTracker:
    """跨帧导弹关联：帧内导弹列表无稳定 ID，按最近邻匹配维持 TacView 对象。"""

    def __init__(self):
        self.active = {}          # id -> (side, e, n, h)
        self._next = {"own": 0x1000, "enemy": 0x2000}

    def update(self, side, missiles, lat0, lon0):
        """返回 (新建行, 更新行, 消失 id 列表)。"""
        unmatched = {i for i, (s, *_rest) in self.active.items() if s == side}
        lines_new, lines_upd, matched = [], [], set()
        for m in missiles:
            e, n, h = m["pos"]
            best, best_d = None, _MSL_MATCH_M
            for i in unmatched:
                _, pe, pn, ph = self.active[i]
                d = math.dist((e, n, h), (pe, pn, ph))
                if d < best_d:
                    best, best_d = i, d
            lon, lat = _enu_to_lonlat(e, n, lat0, lon0)
            if best is None:
                best = self._next[side]
                self._next[side] += 1
                parent, color, name = (
                    (OWN_ID, "Blue", "AAM") if side == "own"
                    else (ENEMY_ID, "Red", "AAM"))
                lines_new.append(
                    f"{best:x},T={lon:.7f}|{lat:.7f}|{h:.1f},Type=Weapon+Missile,"
                    f"Color={color},Parent={parent:x},Name={name}")
            else:
                lines_upd.append(f"{best:x},T={lon:.7f}|{lat:.7f}|{h:.1f}")
            unmatched.discard(best)
            matched.add(best)
            self.active[best] = (side, e, n, h)
        gone = [i for i in unmatched if i not in matched]
        for i in gone:
            del self.active[i]
        return lines_new, lines_upd, gone

    def flush(self, side):
        """末帧冲刷：返回该侧仍在飞导弹 id 并移除（补移除行与射击日志）。"""
        ids = [i for i, (s, *_) in self.active.items() if s == side]
        for i in ids:
            del self.active[i]
        return ids


def export_acmi(frames, path, lat0_deg=30.0, lon0_deg=120.0, title="BVR 1v1"):
    """导出帧序列为 ACMI 2.2 文本文件。"""
    frames = list(frames)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    out = [
        "FileType=text/acmi/tacview",
        "FileVersion=2.2",
        f"0,ReferenceTime={now:%Y-%m-%dT%H:%M:%SZ}",
        f"0,ReferenceLongitude={lon0_deg}",
        f"0,ReferenceLatitude={lat0_deg}",
        "0,DataSource=JSBSim BVRCombatEnv",
        "0,DataRecorder=Loyal-Wingman visualization.acmi",
        f"0,Title={title}",
    ]
    tracker = _MissileTracker()
    locked = {"own": False, "enemy": False}   # 上帧锁定状态（失锁时移除属性）
    n_frames = len(frames)
    for k, frm in enumerate(frames):
        own, ene = frm["own"], frm["enemy"]
        ev = frm.get("events", {})
        last = k == n_frames - 1
        dist_m = frm.get("dist_m") or math.dist(own["pos"], ene["pos"])
        out.append(f"#{frm['t']:.2f}")
        if k == 0:   # 首帧 Bookmark：标注初始距离（双机 id 便于双击定位）
            out.append(f"0,Event=Bookmark|{OWN_ID:x}|{ENEMY_ID:x}"
                       f"|Initial range {dist_m / 1000.0:.1f} km")
        for oid, ac, color, pilot, side in (
                (OWN_ID, own, "Blue", "Agent", "own"),
                (ENEMY_ID, ene, "Red", "ScriptAI", "enemy")):
            e, n, h = ac["pos"]
            lon, lat = _enu_to_lonlat(e, n, lat0_deg, lon0_deg)
            t = (f"{oid:x},T={lon:.7f}|{lat:.7f}|{h:.1f}"
                 f"|{_deg(ac.get('phi_rad', 0.0)):.2f}"
                 f"|{_deg(ac.get('theta_rad', 0.0)):.2f}"
                 f"|{_deg(ac['psi_rad']) % 360.0:.2f}")
            if k == 0:
                coalition = "Allies" if side == "own" else "Enemies"
                t += (f",Name=F-104,Type=Air+FixedWing,Color={color},Pilot={pilot}"
                      f",Coalition={coalition}")
            t += f",TAS={ac['vtrue_mps']:.1f},Mach={ac['mach']:.3f}"
            # ---- 雷达锁定连线：TRACK → LockedTarget；失锁 → 空值移除 ----
            radar_state = (frm.get("radar_state") if side == "own"
                           else frm.get("enemy_radar_state"))
            if radar_state == "TRACK":
                target = ENEMY_ID if side == "own" else OWN_ID
                t += (f",LockedTarget={target:x},LockedTargetMode=1"
                      f",LockedTargetRange={dist_m:.0f}")
                locked[side] = True
            elif locked[side]:
                t += ",LockedTarget=,LockedTargetMode=0"
                locked[side] = False
            # ---- 命中帧：Health=0 触发 TacView 爆炸/残骸显示 ----
            if (side == "enemy" and ev.get("hit_enemy")) or \
                    (side == "own" and ev.get("hit_own")):
                t += ",Health=0"
            out.append(t)
        for side, msls in (("own", frm["own_missiles"]), ("enemy", frm["enemy_missiles"])):
            new, upd, gone = tracker.update(side, msls, lat0_deg, lon0_deg)
            if last:
                gone += tracker.flush(side)   # 末帧在飞导弹：记脱靶并移除
            out.extend(new)
            out.extend(upd)
            out.extend(f"-{i:x}" for i in gone)
            if gone:   # ---- 射击日志（Timeout 事件，进 TacView shot log）----
                src, tgt = ((OWN_ID, ENEMY_ID) if side == "own"
                            else (ENEMY_ID, OWN_ID))
                kill = ev.get("hit_enemy") if side == "own" else ev.get("hit_own")
                for _i in gone:
                    outcome = "Kill" if kill else "Miss"
                    kill = False   # 每帧每侧至多一个命中事件 → 仅首枚记 Kill
                    out.append(f"0,Event=Timeout|SourceId:{src:x}|AmmoType:AAM"
                               f"|AmmoCount:1|TargetId:{tgt:x}|Outcome:{outcome}")
        if ev.get("fired_own"):
            out.append(f"0,Event=Message|{OWN_ID:x}|Blue launches AAM")
        if ev.get("fired_enemy"):
            out.append(f"0,Event=Message|{ENEMY_ID:x}|Red launches AAM")
        if ev.get("hit_enemy"):
            out.append(f"0,Event=Destroyed|{ENEMY_ID:x}")
        if ev.get("hit_own"):
            out.append(f"0,Event=Destroyed|{OWN_ID:x}")
        reason = frm.get("terminated_reason")
        if reason:
            out.append(f"0,Event=Bookmark|Episode ends: {reason}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("﻿" + "\n".join(out) + "\n")
    return path
