# -*- coding: utf-8 -*-
"""
convert_jsbsim_csv.py
=====================
将 JSBSim 原始 CSV 输出转换为项目标准飞行数据 CSV（SI 单位，英制原值并排保留）。

用法:
    py convert_jsbsim_csv.py <原始CSV路径> [输出CSV路径] [重采样率Hz]

示例:
    py convert_jsbsim_csv.py JSBSim/f104_cruise_raw.csv 样例/f104_cruise_si.csv
    py convert_jsbsim_csv.py JSBSim/f104_cruise_raw.csv 样例/f104_cruise_si_1hz.csv 1

说明:
    - JSBSim 1.2.4 的 CSV 列名带 /fdm/jsbsim/ 前缀，本脚本自动去除。
    - 输出文件同时包含 SI 列（如 h_m, vtrue_mps）和原始英制列（如 h_ft, vtrue_fps），
      便于后续训练代码与人工核查双方使用。
    - 重采样率缺省为 0（不重采样，保持原始 10 Hz）。
"""
import sys

import numpy as np
import pandas as pd

FT2M = 0.3048
FPS2MPS = 0.3048
SLUG2KG = 14.59390294

SI_COLUMNS = {
    "Time": "t_s",
    "position/h-sl-ft": "h_sl_m",
    "position/h-agl-ft": "h_agl_m",
    "position/lat-geod-deg": "lat_deg",
    "position/long-gc-deg": "lon_deg",
    "position/distance-from-start-lat-mt": "x_north_m",
    "position/distance-from-start-lon-mt": "x_east_m",
    "velocities/vtrue-fps": "vtrue_mps",
    "velocities/mach": "mach",
    "velocities/u-fps": "u_mps",
    "velocities/v-fps": "v_mps",
    "velocities/w-fps": "w_mps",
    "velocities/v-north-fps": "vn_mps",
    "velocities/v-east-fps": "ve_mps",
    "velocities/v-down-fps": "vd_mps",
    "velocities/p-rad_sec": "p_radps",
    "velocities/q-rad_sec": "q_radps",
    "velocities/r-rad_sec": "r_radps",
    "attitude/phi-rad": "phi_rad",
    "attitude/theta-rad": "theta_rad",
    "attitude/psi-rad": "psi_rad",
    "aero/alpha-rad": "alpha_rad",
    "aero/beta-rad": "beta_rad",
    "accelerations/Nx": "nx_g",
    "accelerations/Ny": "ny_g",
    "accelerations/Nz": "nz_g",
    "fcs/throttle-cmd-norm": "throttle",
    "fcs/elevator-cmd-norm": "elevator",
    "fcs/aileron-cmd-norm": "aileron",
    "fcs/rudder-cmd-norm": "rudder",
    "propulsion/engine/thrust-lbs": "thrust_lb",
    "propulsion/total-fuel-lbs": "fuel_lb",
    "inertia/mass-slugs": "mass_kg",
    "atmosphere/T-R": "temp_K",
    "atmosphere/rho-slugs_ft3": "rho_kgpm3",
}

FT_COLS = {"position/h-sl-ft", "position/h-agl-ft"}
FPS_COLS = {"velocities/vtrue-fps", "velocities/u-fps", "velocities/v-fps",
            "velocities/w-fps", "velocities/v-north-fps", "velocities/v-east-fps",
            "velocities/v-down-fps"}


def load_raw(src):
    """读取 JSBSim 原始 CSV，去除 /fdm/jsbsim/ 列名前缀。"""
    df = pd.read_csv(src)
    df.columns = [c.split("jsbsim/")[-1] if "jsbsim/" in c else c for c in df.columns]
    return df


def build_si_dataframe(df):
    """把原始 DataFrame 转换为标准 SI 列（并保留英制原值列）。"""
    out = pd.DataFrame()
    out["t_s"] = df["Time"]

    for raw, si in SI_COLUMNS.items():
        if raw not in df.columns:
            print(f"[warn] 缺少列: {raw}")
            continue
        vals = df[raw].to_numpy(dtype=float)
        if raw in FT_COLS:
            out[si] = vals * FT2M
            out[raw.replace("/", "_")] = vals
        elif raw in FPS_COLS:
            out[si] = vals * FPS2MPS
            out[raw.replace("/", "_")] = vals
        elif raw == "inertia/mass-slugs":
            out[si] = vals * SLUG2KG
            out[raw.replace("/", "_")] = vals
        elif raw == "atmosphere/rho-slugs_ft3":
            out[si] = vals * 515.3788184  # slugs/ft^3 -> kg/m^3
            out[raw.replace("/", "_")] = vals
        elif raw == "atmosphere/T-R":
            out[si] = vals * 5.0 / 9.0  # Rankine -> Kelvin
            out[raw.replace("/", "_")] = vals
        elif raw == "Time":
            continue
        else:
            out[si] = vals
    return out


def resample(df, hz):
    """最近邻重采样到指定频率。"""
    dt = 1.0 / hz
    t = np.arange(df["t_s"].iloc[0], df["t_s"].iloc[-1] + 1e-9, dt)
    return df.set_index("t_s").reindex(t, method="nearest").reset_index()


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    src = argv[1]
    dst = argv[2] if len(argv) > 2 else src.replace("_raw.csv", "_si.csv")
    resample_hz = float(argv[3]) if len(argv) > 3 else 0.0

    out = build_si_dataframe(load_raw(src))

    if resample_hz > 0:
        out = resample(out, resample_hz)

    out.to_csv(dst, index=False, float_format="%.6f")
    print(f"[ok] 已写出 {dst}  行数={len(out)}  列数={len(out.columns)}")
    print("     列名:", ", ".join(out.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
