# -*- coding: utf-8 -*-
"""
tests/test_radar_model.py（交付物 D2.2-4 单元测试：雷达模型）
============================================================
覆盖：火控雷达 RCS^(1/4) 距离换算、探测→跟踪→丢失→重捕获状态机、
扫描体积（±60°）限制、最大跟踪目标数（10）、RWR 锁定告警与方位量化、
MAWS 逼近告警与 TTA 估计。
运行（DC 环境）：python -m pytest 多任务智能体/tests -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from common.radar_model import FireControlRadar, RWR, MAWS


def _t(tid, x, y, z, rcs=5.0):
    return {"id": tid, "pos": np.array([x, y, z], dtype=float), "rcs": rcs}


# ---------------------------------------------------------------------------
# 火控雷达
# ---------------------------------------------------------------------------
def test_detection_range_rcs_scaling():
    """雷达方程：R_max ∝ RCS^(1/4)——RCS 增大 16 倍 → R_max 增大 2 倍。"""
    r = FireControlRadar()
    base = r.max_detection_range(5.0)
    assert base == pytest.approx(80e3, rel=1e-9)
    assert r.max_detection_range(5.0 * 16) == pytest.approx(2.0 * base, rel=1e-9)
    assert r.max_detection_range(5.0 / 16) == pytest.approx(0.5 * base, rel=1e-9)
    # 典型目标：战斗机 3 m²、大型机 10 m²
    assert r.max_detection_range(3.0) == pytest.approx(80e3 * (3.0 / 5.0) ** 0.25)
    assert r.max_detection_range(10.0) == pytest.approx(80e3 * (10.0 / 5.0) ** 0.25)


def test_detect_track_lost_reacquire():
    """状态机：探测→TRACK，转出扫描体积→LOST，重新进入体积连续 N 次扫描→重捕获 TRACK。"""
    radar = FireControlRadar()
    own = np.zeros(3)
    # 1) 探测并跟踪：正前方 30 km，方位 0°
    radar.update([_t("t1", 30000.0, 0.0, 6000.0)], own, 0.0)
    assert radar.state("t1") == FireControlRadar.TRACK
    assert radar.track_range("t1") == pytest.approx(np.hypot(30000.0, 6000.0))
    # 2) 目标转出方位扫描范围（方位 > 60°）→ 丢失
    for _ in range(3):
        radar.update([_t("t1", 10000.0, 30000.0, 6000.0)], own, 0.0)
    assert radar.state("t1") == FireControlRadar.LOST
    # 3) 目标回到扫描体积内，连续 REACQUIRE_SCANS 次更新后重捕获
    for i in range(FireControlRadar.REACQUIRE_SCANS - 1):
        radar.update([_t("t1", 20000.0, 11547.0, 6000.0)], own, 0.0)
        assert radar.state("t1") == FireControlRadar.LOST, "未满足连续扫描次数前不应重捕获"
    radar.update([_t("t1", 20000.0, 11547.0, 6000.0)], own, 0.0)
    assert radar.state("t1") == FireControlRadar.TRACK


def test_scan_volume_limits():
    """扫描体积限制：方位/俯仰超出 ±60° 不可探测，范围内可探测。"""
    radar = FireControlRadar()
    own = np.zeros(3)
    # 方位 70°：不可探测
    radar.update([_t("a", 10000.0 * np.cos(np.deg2rad(70)), 10000.0 * np.sin(np.deg2rad(70)), 6000.0)], own, 0.0)
    assert radar.state("a") is None
    # 俯仰 70°：不可探测
    radar.update([_t("b", 10000.0 * np.cos(np.deg2rad(70)), 0.0, 10000.0 * np.sin(np.deg2rad(70)))], own, 0.0)
    assert radar.state("b") is None
    # 方位 50°（范围内）：可探测
    radar.update([_t("c", 10000.0 * np.cos(np.deg2rad(50)), 10000.0 * np.sin(np.deg2rad(50)), 6000.0)], own, 0.0)
    assert radar.state("c") == FireControlRadar.TRACK


def test_detection_beyond_range():
    """超出 R_max 的目标不可探测；进入后转为跟踪。"""
    radar = FireControlRadar()
    own = np.zeros(3)
    radar.update([_t("far", 85000.0, 0.0, 6000.0)], own, 0.0)
    assert radar.state("far") is None
    radar.update([_t("far", 60000.0, 0.0, 6000.0)], own, 0.0)
    assert radar.state("far") == FireControlRadar.TRACK


def test_max_track_limit():
    """最大同时跟踪目标数 10：12 个可探测目标只跟踪最近的 10 个。"""
    radar = FireControlRadar()
    own = np.zeros(3)
    targets = [_t(f"t{i}", 5000.0 + 1000.0 * i, 0.0, 6000.0) for i in range(12)]
    radar.update(targets, own, 0.0)
    tracked = radar.tracked_ids
    assert len(tracked) == radar.max_tracks == 10
    expected = [f"t{i}" for i in range(10)]   # 最近的 10 个
    assert sorted(tracked) == sorted(expected)
    # 其余两个为 LOST（曾满足探测但受容量限制未获跟踪）
    assert radar.state("t10") == FireControlRadar.LOST
    assert radar.state("t11") == FireControlRadar.LOST


# ---------------------------------------------------------------------------
# RWR
# ---------------------------------------------------------------------------
def test_rwr_lock_warning_and_quantization():
    """锁定告警：敌雷达 STT 锁定本机 → alarm=True；方位 15° 量化；距离不可测。"""
    rwr = RWR()
    own = np.zeros(3)
    em = [{"id": "e1", "pos": np.array([0.0, 40000.0, 6000.0]),
           "mode": "lock", "lock_target": "self"}]
    r = rwr.update(em, own, 0.0)
    assert r["alarm"] is True
    assert len(r["threats"]) == 1
    th = r["threats"][0]
    assert th["bearing_deg"] == 90.0          # 正右方 → 90°，15° 量化无误差
    assert th["range"] is None                # 无源定位不能测距
    assert th["mode"] == "lock"


def test_rwr_bearing_quantization_accuracy():
    """方位量化：真实方位 37° → 输出 30°（误差 ≤ 半量化间隔 7.5°）。"""
    rwr = RWR()
    own = np.zeros(3)
    az = np.deg2rad(37.0)
    pos = 40000.0 * np.array([np.cos(az), np.sin(az), 0.15])
    r = rwr.update([{"id": "e2", "pos": pos, "mode": "search"}], own, 0.0)
    reported = r["threats"][0]["bearing_deg"]
    assert abs(reported - 37.0) <= rwr.bearing_quant_deg / 2.0 + 1e-9


def test_rwr_sensitivity_and_no_lock():
    """超出灵敏度的辐射源不上报；search 模式不触发锁定告警。"""
    rwr = RWR()
    own = np.zeros(3)
    r = rwr.update([{"id": "far", "pos": np.array([300e3, 0.0, 6000.0]), "mode": "lock",
                     "lock_target": "self"}], own, 0.0)
    assert r["alarm"] is False and not r["threats"]
    r = rwr.update([{"id": "near", "pos": np.array([20000.0, 0.0, 6000.0]), "mode": "search"}],
                   own, 0.0)
    assert r["alarm"] is False
    assert r["threats"][0]["mode"] == "search"


# ---------------------------------------------------------------------------
# MAWS
# ---------------------------------------------------------------------------
def test_maws_approaching_warning_and_tta():
    """来袭导弹：告警 + TTA = r / V_c（解析对照）。"""
    maws = MAWS()
    own_pos = np.zeros(3)
    own_vel = np.zeros(3)
    ms = [{"id": "m1", "pos": np.array([4000.0, 0.0, 0.0]),
           "vel": np.array([-300.0, 0.0, 0.0])}]
    r = maws.update(ms, own_pos, own_vel)
    assert r["alarm"] is True
    w = r["warnings"][0]
    assert w["closing_rate"] == pytest.approx(300.0)
    assert w["tta"] == pytest.approx(4000.0 / 300.0, rel=1e-9)
    assert w["range"] == pytest.approx(4000.0)


def test_maws_no_warning_for_receding_or_far():
    """远离/超探测距离/接近率过低：不告警。"""
    maws = MAWS()
    own_pos = np.zeros(3)
    own_vel = np.zeros(3)
    r = maws.update([{"id": "m1", "pos": np.array([4000.0, 0.0, 0.0]),
                      "vel": np.array([300.0, 0.0, 0.0])}], own_pos, own_vel)   # 远离
    assert not r["warnings"] and r["alarm"] is False
    r = maws.update([{"id": "m2", "pos": np.array([12000.0, 0.0, 0.0]),
                      "vel": np.array([-300.0, 0.0, 0.0])}], own_pos, own_vel)  # 超 8 km
    assert not r["warnings"]
    r = maws.update([{"id": "m3", "pos": np.array([2000.0, 0.0, 0.0]),
                      "vel": np.array([-20.0, 0.0, 0.0])}], own_pos, own_vel)   # 接近率过低
    assert not r["warnings"]
