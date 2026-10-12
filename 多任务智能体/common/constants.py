# -*- coding: utf-8 -*-
"""
多任务智能体/common/constants.py（统一常量层，接口规范 §2.1/§2.5）
================================================================
物理/地理/平台常量的**唯一来源**：禁止在其它模块重复定义（历史重复项见
迁移记录：G0、声速换算、燃油总量、地理原点、经纬度→米换算）。

约定：常量名全大写；数值常量在注释中标注单位。
"""
from __future__ import annotations

import numpy as np

# ---- 单位换算 ----
FT2M = 0.3048                 # 英尺 → 米
M2FT = 1.0 / FT2M
FPS2MPS = 0.3048              # 英尺/秒 → 米/秒
SLUG2KG = 14.59390294
KT2MPS = 0.5144444444

# ---- 物理 ----
G0 = 9.80665                  # 标准重力加速度 m/s²

# ---- 国际标准大气（ISA）----
RHO0 = 1.225                  # 海平面密度 kg/m³
T0 = 288.15                   # 海平面温度 K
L_ISA = -0.0065               # 温度梯度 K/m
R_AIR = 287.05               # 空气气体常数 J/(kg·K)
GAMMA_AIR = 1.4
H_TROP_M = 11000.0            # 对流层顶 m
RHO_TROP = 0.3639175          # 11 km 处密度 kg/m³

# ---- 地理换算与原点（环境 reset 与可视化/ACMI 默认一致）----
METERS_PER_DEG = 111320.0     # 纬度 1° 对应米（简化球面）
GEO_LAT0_DEG = 30.0           # 默认地理原点纬度
GEO_LON0_DEG = 120.0          # 默认地理原点经度

# ---- F-104 平台 ----
F104_FUEL_CAPACITY_LBS = 4700.0   # 标准采集配置总燃油（3500+600+600）


def wrap_pi(a):
    """角度归一化到 [-π, π)（弧度，支持标量/ndarray）。"""
    arr = np.asarray(a, dtype=float)
    out = (arr + np.pi) % (2.0 * np.pi) - np.pi
    return float(out) if out.ndim == 0 else out


def wrap_deg180(a):
    """角度归一化到 [-180, 180)（度，支持标量/ndarray）。"""
    arr = np.asarray(a, dtype=float)
    out = (arr + 180.0) % 360.0 - 180.0
    return float(out) if out.ndim == 0 else out
