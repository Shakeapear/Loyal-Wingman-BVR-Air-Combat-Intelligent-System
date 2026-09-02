# -*- coding: utf-8 -*-
"""
common/jsbsim_env.py
====================
JSBSim 单机飞行 Gym 环境（项目步骤 2.3 JSBSimGymBridge 的任务技能环境，
5 个技能任务的训练/验证接口保持不变）。

2026-08 改造说明（步骤 2.3）：
- JSBSim 生命周期管理（init/load/reset/step/close、标准采集配置）迁移到
  common/jsbsim_bridge.py 的 JSBSimGymBridge 并复用；
- 对外接口（JSBSimFlightEnv 构造、reset/step/观测/动作空间、任务奖励）不变，
  run_validation.py --skip-train 5 任务回归通过。

设计要点（与《05 训练数据采集与样例详解》的 trajectory_v1 对齐）：
- 决策频率 1 Hz，动作 = 高层指令 [速度指令 m/s, 滚转角指令 rad, 高度指令 m]，
  与轨迹文件中的 action_speed_cmd_mps / action_roll_cmd_rad / action_alt_cmd_m 一一对应；
- 内环由本项目为 F-104 新增的自动驾驶仪 autopilot-hold.xml 执行（dt=1/120 s）；
- 观测为标准状态子集（SI 单位，与 47 列字段字典同名列一致）。

使用（DC 环境）：
    env = JSBSimFlightEnv(task_cfg)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""
import gymnasium as gym
import numpy as np

from common.jsbsim_bridge import JSBSimGymBridge, SIM_DT

DECISION_HZ = 1.0             # 决策频率（Hz）
FRAMES_PER_STEP = int(round(DECISION_HZ / SIM_DT))

ACTION_LOW = np.array([180.0, -0.90, 3000.0])   # [v_cmd m/s, roll_cmd rad, alt_cmd m]
ACTION_HIGH = np.array([280.0, 0.90, 8000.0])

OBS_FIELDS = ["h_sl_m", "vtrue_mps", "mach", "phi_rad", "theta_rad", "psi_rad",
              "alpha_rad", "beta_rad", "nz_g", "throttle", "fuel_frac", "vd_mps"]


class KinematicTarget:
    """虚拟目标（多机态势的占位实现：常速质点在水平面做圆周运动）。

    后期由步骤 2.2 的多机态势模块替换为第二架 JSBSim 飞机。
    """

    def __init__(self, x0=3000.0, y0=-2000.0, v=180.0, turn_rate=0.03, h0=6000.0):
        self.x, self.y, self.h = x0, y0, h0
        self.v = v
        self.turn_rate = turn_rate
        self.heading = np.pi  # 朝我机方向飞来

    def step(self, dt=1.0):
        self.heading += self.turn_rate * dt
        self.x += self.v * np.cos(self.heading) * dt
        self.y += self.v * np.sin(self.heading) * dt


class JSBSimFlightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task_cfg: dict):
        super().__init__()
        self.cfg = task_cfg
        self.max_steps = task_cfg.get("max_steps", 120)

        self.bridge = JSBSimGymBridge(aircraft="f104")
        self.fdm = self.bridge.fdm

        n_extra = len(task_cfg.get("extra_obs", []))
        n_obs = len(OBS_FIELDS) + n_extra
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        if task_cfg.get("discrete_action"):
            self.action_space = gym.spaces.Discrete(task_cfg["n_discrete"])
        else:
            self.action_space = gym.spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)

        self.target = None
        if task_cfg.get("use_target"):
            self.target = KinematicTarget(**task_cfg.get("target_kwargs", {}))

    # ---------- 内部辅助 ----------

    def _set(self, name, value):
        self.fdm.set_property_value(name, value)

    def _get(self, name):
        return self.fdm.get_property_value(name)

    def _setup_aircraft(self):
        """标准采集配置（与《04 数据采集规范》§3 标准事件块一致）+ 接通自驾仪。"""
        self.bridge.setup_standard(autopilot=True)

    def _observe(self):
        s = self.bridge.state_dict()
        obs = [
            s["h_sl_m"],
            s["vtrue_mps"],
            s["mach"],
            s["phi_rad"],
            s["theta_rad"],
            s["psi_rad"],
            s["alpha_rad"],
            s["beta_rad"],
            s["nz_g"],
            s["throttle"],
            s["fuel_frac"],
            s["vd_mps"],
        ]
        for field in self.cfg.get("extra_obs", []):
            obs.append(self._extra_obs_field(field))
        return np.array(obs, dtype=np.float32)

    def _extra_obs_field(self, field):
        if field == "rel_north_m":
            return self.target.y
        if field == "rel_east_m":
            return self.target.x
        if field == "rel_alt_m":
            return self.target.h - self._get("position/h-sl-ft") * 0.3048
        if field == "dist_m":
            return float(np.hypot(self.target.x, self.target.y))
        if field == "bearing_rad":
            return float(np.arctan2(self.target.y, self.target.x))
        raise ValueError(f"未知观测字段 {field}")

    def _apply_action(self, action):
        """连续动作 -> 自驾仪设定；离散动作 -> 查表转换为设定。"""
        if self.cfg.get("discrete_action"):
            action = self._decode_discrete(action)
        v_cmd, roll_cmd, alt_cmd = np.clip(action, ACTION_LOW, ACTION_HIGH)
        self._last_ap_cmds = (float(roll_cmd), float(alt_cmd) / 0.3048,
                              float(v_cmd) / 0.3048)   # (roll_rad, alt_ft, speed_fps)
        self._set("fcs/ap-speed-setpoint-fps", self._last_ap_cmds[2])
        self._set("fcs/ap-roll-setpoint-rad", self._last_ap_cmds[0])
        self._set("fcs/ap-alt-setpoint-ft", self._last_ap_cmds[1])

    def _decode_discrete(self, a):
        """离散动作：a = heading_bin * n_throttle + throttle_bin（默认 16 航向 × 3 油门）。"""
        n_throttle = self.cfg["n_throttle"]
        heading_bin, throttle_bin = divmod(int(a), n_throttle)
        n_heading = self.cfg["n_heading"]
        heading = -np.pi + 2 * np.pi * heading_bin / n_heading
        v_cmd = [200.0, 230.0, 260.0][throttle_bin]
        # 滚转角指令 = 航向误差的比例控制（替代自驾仪航向环）
        roll_cmd = np.clip(0.8 * float(np.arctan2(np.sin(heading - self._get("attitude/psi-rad")),
                                                 np.cos(heading - self._get("attitude/psi-rad")))),
                           -0.9, 0.9)
        alt_cmd = self._get("position/h-sl-ft") * 0.3048
        return np.array([v_cmd, roll_cmd, alt_cmd])

    def _task_reward(self):
        return self.cfg["reward_fn"](self)

    def _terminated(self):
        h_agl = self._get("position/h-agl-ft") * 0.3048
        if h_agl < 30.0:
            return True
        if abs(self._get("attitude/phi-rad")) > 2.6:
            return True
        return self.cfg.get("done_fn", lambda env: False)(self)

    # ---------- Gym 接口 ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        # 固定巡航初始条件（15000 ft / M0.85 / 航向 0°），与数据采集一致
        self.bridge.reset(seed=seed, h_ft=15000.0, mach=0.85, heading_deg=0.0,
                          autopilot=True)
        self.fdm = self.bridge.fdm
        if self.target is not None:
            kw = self.cfg.get("target_kwargs", {})
            self.target.__init__(**kw)
        from common.tasks import init_task_state
        init_task_state(self)
        return self._observe(), {}

    def step(self, action):
        self._apply_action(action)
        self.bridge.step_ap(*self._last_ap_cmds, FRAMES_PER_STEP)
        if self.target is not None:
            self.target.step(1.0)
        self.steps += 1
        obs = self._observe()
        reward = self._task_reward()
        terminated = self._terminated()
        truncated = self.steps >= self.max_steps
        info = {"steps": self.steps,
                "h_sl_m": obs[0], "vtrue_mps": obs[1], "reward": float(reward)}
        if terminated:
            info["crash"] = self._get("position/h-agl-ft") * 0.3048 < 30.0
        return obs, float(reward), terminated, truncated, info
