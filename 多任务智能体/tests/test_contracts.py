# -*- coding: utf-8 -*-
"""
多任务智能体/tests/test_contracts.py（接口契约基线测试，接口规范 §2.2/§2.4）
================================================================
把"当前契约"固化为可执行基线，后续任何字段增删/改名都会在此暴露：

- 常量单一来源（common/constants）；
- 帧字典契约：FRAME_KEYS_V1 与 BVRCombatEnv.get_viz_frame() 实测键集**完全一致**；
- info 契约：reset 与 step 键集均覆盖 INFO_KEYS_STEP；
- 技能级 info 契约：JSBSIM_INFO_KEYS；
- 未 reset 取帧报 ExecutionError(code=ENV_NOT_RESET)（RuntimeError 兼容）；
- 旧键迁移 migrate_frame_keys；
- 单局结果记录 EpisodeResult 版本字段；
- 异常层级/退出码；命名与路径安全工具。

运行（DC 环境，在 多任务智能体/ 目录下）：python -m pytest tests/test_contracts.py -v
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from common import constants, contracts
from common.bvr_combat_env import BVRCombatEnv
from common.errors import (ENV_NOT_RESET, EXIT_BIZ, EXIT_CONFIG, EXIT_EXTERNAL,
                           ConfigError, DataFormatError, ExecutionError,
                           ExternalServiceError, ProjectError, ValidationError,
                           exit_code_for)
from common.tasks import TASKS
from common.jsbsim_env import JSBSimFlightEnv

NOOP = {"flight": np.array([0.55, 0.0, 0.0, 0.0], dtype=np.float32),
        "weapon": np.array([0, 0, 0])}


@pytest.fixture(scope="module")
def env():
    e = BVRCombatEnv(config={"compute_envelope": False})
    e.reset(seed=7)
    yield e
    e.close()


class TestConstants:
    def test_single_source_values(self):
        assert constants.G0 == pytest.approx(9.80665)
        assert constants.FT2M == pytest.approx(0.3048)
        assert constants.METERS_PER_DEG == pytest.approx(111320.0)
        assert constants.F104_FUEL_CAPACITY_LBS == pytest.approx(4700.0)

    def test_wrappers(self):
        assert constants.wrap_pi(3.5) == pytest.approx(3.5 - 2 * np.pi)
        assert constants.wrap_deg180(190.0) == pytest.approx(-170.0)
        assert constants.wrap_deg180(np.array([10.0, 350.0])).tolist() == [10.0, -10.0]


class TestFrameContract:
    def test_frame_keys_exactly_match_contract(self, env):
        frame = env.get_viz_frame()
        assert set(frame.keys()) == set(contracts.FRAME_KEYS_V1)
        assert contracts.validate_frame(frame) == []

    def test_frame_accepts_generated_episode(self, env):
        env.step(NOOP)
        frame = env.get_viz_frame()
        assert contracts.validate_frame(frame) == []
        # 单位后缀契约：半径/时间带单位，航向/方位为 rad
        assert "r_max_own_m" in frame and "maws_tta_s" in frame
        assert "rwr_bearing_rad" in frame

    def test_requires_reset(self):
        e = BVRCombatEnv(config={"compute_envelope": False})
        try:
            with pytest.raises(ExecutionError) as ei:
                e.get_viz_frame()
            assert ei.value.code == ENV_NOT_RESET
            assert isinstance(ei.value, RuntimeError)   # 兼容旧 except RuntimeError
        finally:
            e.close()

    def test_migrate_legacy_keys(self):
        legacy = {"r_max_own": 1.0, "maws_tta": 2.0, "rwr_bearing": 3.0}
        out = contracts.migrate_frame_keys(dict(legacy))
        assert out["r_max_own_m"] == 1.0 and out["maws_tta_s"] == 2.0
        assert out["rwr_bearing_rad"] == 3.0


class TestInfoContract:
    def test_reset_and_step_keys(self, env):
        _, info0 = env.reset(seed=11)
        assert contracts.validate_info(info0) == []
        _, _, _, _, info1 = env.step(NOOP)
        assert contracts.validate_info(info1) == []
        # 弹数概念区分：info 用 *_left，帧用列表
        assert "own_missiles_left" in info1 and "own_missiles" not in info1
        assert info1["r_max_enemy_m"] is not None

    def test_skill_env_info_keys(self):
        e = JSBSimFlightEnv(TASKS["cruise_hold"])
        try:
            _, info0 = e.reset(seed=3)
            _, _, _, _, info1 = e.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            for info in (info0, info1):
                missing = [k for k in contracts.JSBSIM_INFO_KEYS if k not in info]
                assert missing == []
            assert isinstance(info1["crash"], bool)   # crash 恒存在（不再只在终止时出现）
        finally:
            e.close()


class TestResultRecord:
    def test_versions_present(self):
        d = contracts.EpisodeResult(seed=1, steps=10).to_dict()
        assert d["record_version"] == contracts.RESULT_RECORD_VERSION
        for key in ("frame", "csv", "info", "result_record", "test_case"):
            assert key in d["schema_versions"]
        assert d["steps"] == 10 and d["logic_errors"] == 0


class TestErrorsAndTools:
    def test_hierarchy_and_exit_codes(self):
        assert issubclass(ProjectError, RuntimeError)
        assert issubclass(ConfigError, ValueError)
        assert issubclass(ValidationError, ValueError)
        assert exit_code_for(ConfigError("CFG_INVALID", "x")) == EXIT_CONFIG
        assert exit_code_for(ValidationError("ARG_INVALID", "x")) == EXIT_CONFIG
        assert exit_code_for(ExternalServiceError("LLM_HTTP_ERROR", "x")) == EXIT_EXTERNAL
        assert exit_code_for(DataFormatError("DATA_BAD_COLUMN", "x")) == EXIT_BIZ
        assert exit_code_for(ExecutionError("ENV_NOT_RESET", "x")) == EXIT_BIZ

    def test_error_message_contains_code(self):
        e = DataFormatError("DATA_BAD_COLUMN", "缺列", detail={"missing": ["a"]})
        assert "[DATA_BAD_COLUMN]" in str(e) and e.to_dict()["detail"]["missing"] == ["a"]

    def test_tag_and_path_tools(self, tmp_path):
        assert contracts.sanitize_tag("../a b") == "a_b"
        assert contracts.sanitize_tag("") == "run"
        assert contracts.sanitize_tag("--", fallback="seed1") == "seed1"
        first = contracts.unique_dir(tmp_path, "run")
        first.mkdir()
        assert contracts.unique_dir(tmp_path, "run").name == "run_1"
        inside = contracts.ensure_within(tmp_path / "a.csv", tmp_path)
        assert inside.name == "a.csv"
        with pytest.raises(ExecutionError):
            contracts.ensure_within(tmp_path.parent / "other.csv", tmp_path)
