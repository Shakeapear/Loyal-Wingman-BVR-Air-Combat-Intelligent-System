# -*- coding: utf-8 -*-
"""
tests/test_visualization.py（步骤 2.4 单元测试：可视化接口与回放链路）
======================================================================
覆盖：get_viz_frame 快照模式（键集/类型/几何一致性）、事件标志、
render("rgb_array") 离屏帧、EpisodeRecorder CSV 往返、render_episode GIF、
ACMI 导出文件头。离屏用例全部走 Agg canvas，无需显示设备。
运行（DC 环境）：python -m pytest 多任务智能体/tests/test_visualization.py -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from common.bvr_combat_env import BVRCombatEnv
from visualization.acmi import export_acmi
from visualization.csv2acmi import convert as csv_to_acmi
from visualization.offscreen import frame_to_rgb, render_episode
from visualization.recorder import EpisodeRecorder
from visualization.tactical_core import TrailHistory, draw_tactical_frame

NOOP = {"flight": np.array([0.55, 0.0, 0.0, 0.0], dtype=np.float32),
        "weapon": np.array([0, 0, 0])}

VIZ_KEYS = {"steps", "t", "own", "enemy", "own_missiles", "enemy_missiles",
            "dist_m", "radar_state", "enemy_radar_state", "rwr_alarm", "rwr_bearing",
            "maws_alarm", "maws_tta", "in_zone", "r_max_own", "r_min_own",
            "r_nez_own", "r_max_enemy", "r_min_enemy", "reward",
            "terminated_reason", "events"}


@pytest.fixture(scope="module")
def env():
    e = BVRCombatEnv(config={"compute_envelope": False})
    e.reset(seed=42)
    yield e
    e.close()


def _run_steps(env, n):
    for _ in range(n):
        env.step(NOOP)


class TestVizFrame:
    def test_schema_and_types(self, env):
        frm = env.get_viz_frame()
        assert VIZ_KEYS <= set(frm.keys())
        assert isinstance(frm["steps"], int) and isinstance(frm["t"], float)
        for side in ("own", "enemy"):
            ac = frm[side]
            assert len(ac["pos"]) == 3 and all(isinstance(v, float) for v in ac["pos"])
            for k in ("psi_rad", "phi_rad", "theta_rad", "vtrue_mps", "mach", "h_sl_m"):
                assert isinstance(ac[k], float)
        assert isinstance(frm["own_missiles"], list)
        assert isinstance(frm["enemy_missiles"], list)
        assert set(frm["events"].keys()) == {"fired_own", "fired_enemy",
                                             "hit_enemy", "hit_own"}

    def test_geometry_consistency(self, env):
        """dist_m 与双方位置一致；敌机绝对位置含初始偏移（初始距离 30~80 km）。"""
        env.reset(seed=42)
        frm = env.get_viz_frame()
        rel = np.subtract(frm["enemy"]["pos"], frm["own"]["pos"])
        assert np.isclose(np.linalg.norm(rel), frm["dist_m"], rtol=1e-6)
        assert 25e3 < frm["dist_m"] < 85e3

    def test_events_flags_after_steps(self, env):
        _run_steps(env, 3)
        frm = env.get_viz_frame()
        assert all(isinstance(v, bool) for v in frm["events"].values())
        assert frm["steps"] == env.steps

    def test_frame_serializable(self, env):
        import json
        frm = env.get_viz_frame()
        json.dumps(frm)   # 纯 Python 标量/list，可 JSON 序列化（记录/回放前提）


class TestRender:
    def test_metadata_render_modes(self):
        assert BVRCombatEnv.metadata["render_modes"] == ["human", "rgb_array"]

    def test_rgb_array_render(self):
        e = BVRCombatEnv(config={"compute_envelope": False, "render_mode": "rgb_array"})
        e.reset(seed=1)
        arr = e.render()
        assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
        assert arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[0] > 100
        e.close()

    def test_render_none_mode_noop(self, env):
        assert env.render_mode is None
        assert env.render() is None     # 训练路径零开销

    def test_viz_frame_requires_reset(self):
        """reset 前 get_viz_frame 应显式报错（而非返回退化快照）。"""
        e = BVRCombatEnv(config={"compute_envelope": False})
        with pytest.raises(RuntimeError, match="reset"):
            e.get_viz_frame()
        e.close()

    def test_display_reset_on_env_reset(self):
        """env.reset() 应重置已挂接的 TacticalDisplay（跨 episode 不残留轨迹）。"""
        e = BVRCombatEnv(config={"compute_envelope": False, "render_mode": "human"})
        calls = []

        class _DummyDisplay:      # 不创建真实 Tk 窗口的替身
            def reset(self):
                calls.append("reset")

            def update(self, frame):
                pass

            def close(self):
                pass

        e._display = _DummyDisplay()
        e.reset(seed=7)
        assert calls == ["reset"]
        e.close()


class TestRecorder:
    def test_csv_roundtrip(self, env, tmp_path):
        rec = EpisodeRecorder()
        _run_steps(env, 5)
        for _ in range(3):
            rec.capture_env(env)
            env.step(NOOP)
        path = rec.save_csv(tmp_path / "ep.csv")
        frames = EpisodeRecorder.load_csv(path)
        assert len(frames) == len(rec) == 3
        f0, r0 = frames[0], rec.frames[0]
        assert f0["steps"] == r0["steps"]
        assert np.isclose(f0["dist_m"], r0["dist_m"])
        assert np.allclose(f0["own"]["pos"], r0["own"]["pos"])
        assert set(f0.keys()) == set(r0.keys())

    def test_csv_has_missile_columns(self, env, tmp_path):
        rec = EpisodeRecorder()
        rec.capture_env(env)
        path = rec.save_csv(tmp_path / "ep.csv")
        header = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert "own_missiles_json" in header and "events_json" in header


class TestOffscreen:
    def test_frame_to_rgb(self, env):
        arr = frame_to_rgb(env.get_viz_frame(), dpi=60)
        assert arr.dtype == np.uint8 and arr.shape[2] == 3

    def test_render_episode_gif(self, env, tmp_path):
        frames = [env.get_viz_frame()]
        env.step(NOOP)
        frames.append(env.get_viz_frame())
        out = render_episode(frames, tmp_path / "ep.gif", fps=5, dpi=60)
        assert out.exists() and out.stat().st_size > 0
        from PIL import Image
        with Image.open(out) as im:
            assert getattr(im, "n_frames", 1) == 2

    def test_draw_tactical_frame_core(self, env):
        """绘图内核冒烟：直接构造 Figure 调 draw_tactical_frame（告警/面板/扇形）。"""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        frm = env.get_viz_frame()
        frm["rwr_alarm"] = True
        frm["maws_alarm"] = True
        hist = TrailHistory()
        hist.push(frm)
        fig = Figure(figsize=(8, 4), dpi=50)
        FigureCanvasAgg(fig)
        am, ap = fig.subplots(1, 2)
        draw_tactical_frame(am, ap, frm, hist, cum_reward=1.0)
        fig.canvas.draw()
        assert np.asarray(fig.canvas.buffer_rgba()).size > 0


def _syn_ac(pos):
    """合成帧飞机字典（export_acmi 所需最小字段）。"""
    return {"pos": list(pos), "psi_rad": 0.0, "phi_rad": 0.0, "theta_rad": 0.0,
            "vtrue_mps": 250.0, "mach": 0.800}


def _syn_frame(t, own_pos, ene_pos, own_m=(), ene_m=(), rs=None, ers=None,
               ev=None, reason=None):
    """合成 viz 帧（纯字典，无需环境，确定性覆盖 ACMI 增强逻辑）。"""
    return {"steps": int(t), "t": float(t),
            "own": _syn_ac(own_pos), "enemy": _syn_ac(ene_pos),
            "own_missiles": [dict(m) for m in own_m],
            "enemy_missiles": [dict(m) for m in ene_m],
            "dist_m": float(np.linalg.norm(np.subtract(ene_pos, own_pos))),
            "radar_state": rs, "enemy_radar_state": ers,
            "events": ev or {"fired_own": False, "fired_enemy": False,
                             "hit_enemy": False, "hit_own": False},
            "terminated_reason": reason}


def _frame_block(text, t):
    """截取 ACMI 文本中 #t 时间帧到下一时间帧之间的行块。"""
    lines = text.splitlines()
    i = lines.index(f"#{t:.2f}")
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("#")),
             len(lines))
    return lines[i:j]


class TestAcmi:
    def test_export_acmi(self, env, tmp_path):
        frames = [env.get_viz_frame()]
        env.step(NOOP)
        frames.append(env.get_viz_frame())
        out = export_acmi(frames, tmp_path / "ep.acmi")
        text = out.read_text(encoding="utf-8")
        assert "FileType=text/acmi/tacview" in text
        assert "FileVersion=2.2" in text
        assert "Type=Air+FixedWing" in text
        assert text.count("#") >= 2          # 两个时间帧

    def test_coalition_and_title(self, tmp_path):
        """首帧红蓝阵营 Coalition=Allies/Enemies；Title 透传（demo 带 seed）。"""
        frames = [_syn_frame(0, [0, 0, 6000], [0, 40000, 6000]),
                  _syn_frame(1, [0, 100, 6000], [0, 40100, 6000])]
        text = export_acmi(frames, tmp_path / "c.acmi",
                           title="BVR 1v1 (seed 8)").read_text(encoding="utf-8")
        assert "0,Title=BVR 1v1 (seed 8)" in text
        assert text.count("Coalition=Allies") == 1     # 仅首帧声明
        assert text.count("Coalition=Enemies") == 1
        own0 = next(l for l in _frame_block(text, 0.0) if l.startswith("100,"))
        ene0 = next(l for l in _frame_block(text, 0.0) if l.startswith("200,"))
        assert "Coalition=Allies" in own0 and "Coalition=Enemies" in ene0
        assert "Event=Bookmark|100|200|Initial range 40.0 km" in text

    def test_locked_target_track_and_drop(self, tmp_path):
        """TRACK 帧双方 LockedTarget 互相指向；失锁帧空值移除 + Mode=0。"""
        frames = [
            _syn_frame(0, [0, 0, 6000], [0, 40000, 6000], rs="TRACK", ers="TRACK"),
            _syn_frame(1, [0, 100, 6000], [0, 40100, 6000], rs="TRACK", ers="TRACK"),
            _syn_frame(2, [0, 200, 6000], [0, 40200, 6000], rs="LOST", ers="LOST"),
        ]
        text = export_acmi(frames, tmp_path / "l.acmi").read_text(encoding="utf-8")
        own1 = next(l for l in _frame_block(text, 1.0) if l.startswith("100,"))
        ene1 = next(l for l in _frame_block(text, 1.0) if l.startswith("200,"))
        assert "LockedTarget=200" in own1 and "LockedTargetMode=1" in own1
        assert "LockedTargetRange=40000" in own1
        assert "LockedTarget=100" in ene1
        own2 = next(l for l in _frame_block(text, 2.0) if l.startswith("100,"))
        ene2 = next(l for l in _frame_block(text, 2.0) if l.startswith("200,"))
        assert "LockedTarget=," in own2 and "LockedTargetMode=0" in own2
        assert "LockedTarget=," in ene2

    def test_hit_frame_health_and_shot_kill(self, tmp_path):
        """命中帧：被命中方 Health=0 + Destroyed；消失导弹记 Outcome:Kill。"""
        msl = {"pos": [0, 500, 6000], "vel": [0, 800, 0], "t": 1.0}
        ev_hit = {"fired_own": True, "fired_enemy": False,
                  "hit_enemy": True, "hit_own": False}
        frames = [
            _syn_frame(0, [0, 0, 6000], [0, 30000, 6000], rs="TRACK"),
            _syn_frame(1, [0, 0, 6000], [0, 29900, 6000], own_m=[msl], rs="TRACK",
                       ev={"fired_own": True, "fired_enemy": False,
                           "hit_enemy": False, "hit_own": False}),
            _syn_frame(2, [0, 0, 6000], [0, 29800, 6000], ev=ev_hit,
                       reason="enemy_hit"),
        ]
        text = export_acmi(frames, tmp_path / "h.acmi").read_text(encoding="utf-8")
        blk = _frame_block(text, 2.0)
        ene2 = next(l for l in blk if l.startswith("200,"))
        assert "Health=0" in ene2                        # 爆炸效果
        assert "0,Event=Destroyed|200" in blk
        assert any(l.startswith("-1000") for l in blk)   # 命中弹移除
        assert ("0,Event=Timeout|SourceId:100|AmmoType:AAM|AmmoCount:1"
                "|TargetId:200|Outcome:Kill") in blk

    def test_missile_miss_shot_log(self, tmp_path):
        """脱靶场景：导弹无命中消失 → Outcome:Miss；末帧在飞弹冲刷也记 Miss。"""
        m1 = {"pos": [0, 500, 6000], "vel": [0, 800, 0], "t": 1.0}
        m2 = {"pos": [0, 29500, 6000], "vel": [0, -800, 0], "t": 1.0}
        frames = [
            _syn_frame(0, [0, 0, 6000], [0, 30000, 6000]),
            _syn_frame(1, [0, 0, 6000], [0, 29900, 6000],
                       own_m=[m1], ene_m=[m2]),
            _syn_frame(2, [0, 0, 6000], [0, 29800, 6000]),   # 双方导弹均无命中消失
        ]
        text = export_acmi(frames, tmp_path / "m.acmi").read_text(encoding="utf-8")
        assert ("0,Event=Timeout|SourceId:100|AmmoType:AAM|AmmoCount:1"
                "|TargetId:200|Outcome:Miss") in text
        assert ("0,Event=Timeout|SourceId:200|AmmoType:AAM|AmmoCount:1"
                "|TargetId:100|Outcome:Miss") in text
        assert "Outcome:Kill" not in text


class TestCsv2Acmi:
    def test_csv_to_acmi(self, env, tmp_path):
        """CSV 录制文件 → ACMI 转换器：输出同名 .acmi 且含增强字段。"""
        rec = EpisodeRecorder()
        _run_steps(env, 2)
        for _ in range(3):
            rec.capture_env(env)
            env.step(NOOP)
        csv_path = rec.save_csv(tmp_path / "ep.csv")
        out = csv_to_acmi(csv_path)
        assert out == tmp_path / "ep.acmi" and out.exists()
        text = out.read_text(encoding="utf-8")
        assert "FileType=text/acmi/tacview" in text
        assert "Coalition=Allies" in text and "0,Title=ep" in text
