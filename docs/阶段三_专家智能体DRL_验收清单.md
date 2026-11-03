# 阶段三专家智能体 DRL 底座验收清单

> 日期：2026-10-30｜对应任务书验收标准｜结论：**本轮最低交付全部达成；assisted 240 步稳定飞行 24/24 通过；完整 JSRL/多机/性能指标明确列为后续项。**

## 1. 逐项验收

### 1.1 任务级智能体

- [x] **4 个任务级智能体均有可训练实现**
  - `StrikeAgent`/`PatrolAgent`/`DefenseAgent`/`EWAgent`，统一继承 `TaskDRLExpertAgent`；
  - 证据：`多任务智能体/agents/task_agents/*.py`、`tests/test_expert_agents.py`（4 个参数化训练用例通过）。
- [x] **4 个任务级智能体均注册到统一接口**
  - `DEFAULT_REGISTRY` 共 9 项（4 task + 5 skill），`tier/action_channel` 可查询；
  - 证据：`agents/expert_registry.py`、`test_registry_default_nine_agents`。
- [x] **5 个技能智能体不退化**
  - 未修改 `base_agent.py` 与 5 个技能智能体类；`cruise_hold` 抽查复跑通过；三套测试全绿；
  - 证据：`validation/results/validation_report.txt`（原 5/5 报告已恢复）、`test_skill_agent_command_v1_channel`。

### 1.2 实例化与动作正确性

- [x] **每个任务级智能体可实例化**
  - 4 个类均以 `env_config/train_config/seed/model_path` 构造；构造本身不启动 JSBSim（惰性建环境）；
  - 证据：`test_task_agent_instantiate_and_action_interface`（参数化 4 任务）。
- [x] **小规模训练无异常**
  - 每个任务 512 步 PPO，4/4 通过；脚本总耗时 8.861 s；
  - 证据：`validation/results/stage3/stage3_expert_validation.txt`。
- [x] **奖励/损失为有限值**
  - 4 个智能体 `all_finite=True`，评估奖励有限；
  - 证据：同上 JSON 的 `train.all_finite`、`eval.reward_finite`。
- [x] **`get_action` 输出 `(7,) float32`，范围符合 ActionV1**
  - 形状/ dtype / 上下界检查全部通过；
  - 证据：`test_task_agent_instantiate_and_action_interface`、验证报告 `action_check`。
- [x] **保存加载后推理一致**
  - 4/4 `load_consistent=True`，最大绝对差 0.0；
  - 证据：验证 JSON 的 `load_action_diff`；测试 `test_task_agent_tiny_train_save_load`。

### 1.3 行为树与兼容通道

- [x] **行为树可按任务名调用全部 4 个任务级智能体**
  - `ExpertMissionSelector` + `MissionIs` + `ExpertAgentControl`，支持英文任务名/中文别名；
  - 测试用例用假环境一次调用 4 任务全部 `SUCCESS`；
  - 真实演示用 `BVRCombatEnv` 依次运行 4 任务，全部 `SUCCESS`；
  - 证据：`test_bt_selects_all_four_task_agents_by_name`、`docs/assets/stage3_expert_demo/demo_summary.json`。
- [x] **演示至少 2 个不同智能体完成最小闭环**
  - 实际演示 4 个：`strike/patrol/defense/ew`，每任务 12 步，累计奖励分别为 60.105 / 0.100 / 18.544 / 60.344；
  - 证据：`docs/assets/stage3_expert_demo/`（CSV/JSON/PNG）。
- [x] **至少 1 个技能智能体通过兼容适配器可调用**
  - `cruise_hold` 可实例化/推理，输出 3 维 AP 指令；`to_ap_command`/`command_v1` 可转换 ActionV1 → 3 维；
  - 证据：`test_skill_agent_command_v1_channel`、`agents/action_adapter.py`。

### 1.4 测试与回归

- [x] **相关 pytest 全部通过**
  - 新增 24 项通过；多任务 80 项、行为树 19 项、可视化 23 项，合计 **122 passed / 0 failed**；
  - 证据：测试报告 §5。
- [x] **新增测试覆盖接口、适配层、注册表、各智能体**
  - 证据：`tests/test_expert_agents.py`（接口、适配、掩码、control_mode、注册表、4 智能体、BT、技能兼容）。

### 1.5 稳定飞行基线（本轮新增硬验收）

- [x] **ActionV1 形状/范围/边沿位不变**
  - 仍为 `(7,) float32`，low/high 与 edge_dims `[4,5,6]` 未改；`raw_fcs`/`assisted` 只改变连续 4 维下游解释；
  - 证据：`test_action_spec_and_command_spec`、`agents/policy_interface.py`。
- [x] **本机 AP 内环可配置（`control_mode=assisted`）**
  - `BVRCombatEnv` 支持 `raw_fcs/assisted`；`MissionMixin` 默认 assisted；训练/部署/行为树共用同一映射；
  - 证据：`common/control_mode.py`、`common/bvr_combat_env.py`、`mission_env.py`、`test_bvr_control_mode_switch_and_240_step_stability`。
- [x] **4/4 任务 240 步隔离飞行无 `own_crash`**
  - 4 任务 × 3 seeds（7/17/27）× safe/model = **24/24 通过**，全部 240 步 `timeout`；
  - 证据：`validation/results/stage3_stability/stability_report.txt`、`docs/阶段三_稳定飞行基线_实现与测试报告.md`。
- [x] **raw_fcs 仍可用（对照）**
  - `BVRCombatEnv` 默认 `raw_fcs`；strike/seed42 复现 45 步 `own_crash`，证明根因与修复有效；
  - 证据：`validation/results/stage3_diagnosis/`。
- [x] **`action_spec` 写明 control_mode 与映射**
  - 新增 `control_mode/control_modes/assisted_mapping/continuous_semantics`；`*.spec.json` 随模型保存；
  - 证据：`action_spec()` 单测、验证脚本保存的 spec。

### 1.6 P1 混合动作网络（2026-11-01）

- [x] **混合动作网络实现**
  - `HybridActorCritic`：FC(256→128) 主干、连续 4 维 μ/log_std、武器 3 维 logits、Critic；
  - `HybridDistribution`：Gaussian+tanh（含雅可比） × Bernoulli；`log_prob`/熵与解析式对拍（atol 1e-5）；
  - 证据：`agents/hybrid_policy.py`、`tests/test_hybrid_policy.py`（14 项）。
- [x] **动作掩码训练/推理一致**
  - 复用 `compute_action_mask`；`MaskedObsWrapper` 将 7 维掩码拼到 obs（28→35）；
  - 非法 fire logits≤−1e7、采样不选、梯度≈0；无弹/未跟踪/超包线/合法发射与 `_try_fire_own` 对拍；
  - 证据：`test_illegal_fire_probability_and_gradient_zero`、`TestMaskParity`。
- [x] **SB3 接入可训**
  - `HybridActorCriticPolicy` 覆写 `_build/forward/evaluate_actions/get_distribution/predict_values`；
  - `python -m agents.ppo_agent --task strike --policy hybrid --timesteps 2048 --n-envs 2 --seed 0` 通过；
  - 2048 步：`ep_rew_mean=304.40`、`policy_gradient_loss=-9.69e-4`、`value_loss=8546.59`、`entropy_loss=-5.704`、`all_finite=True`；评估 `finite/in_bounds=True`；
  - 证据：`models/hybrid_smoke/train_summary.json`（不入库）、`docs/阶段三_混合动作PPO网络_实现与测试报告.md`。
- [x] **保存/加载与推理协议**
  - save/load 后 deterministic 动作一致；`HybridPPOAgent.get_action` 返回 `PolicyOutput(source="drl", masked, action_version="action_v1")`；
  - 证据：`test_learn_save_load_deterministic`、`test_agent_get_action_policy_output`。
- [x] **回归**
  - 三套全量 **187 passed / 0 failed**（含新增路径/掩码/模型规格/边沿触发测试）；原有 MlpPolicy 路径未改。
- [ ] **P1 性能验收（>60% 命中率）**：未做长训练，状态 🟡；留待服务器训练与固定测试集评估。

### 1.7 范围约束

- [x] **未启动完整训练**
  - 仅 512 步/智能体（验证）与 32 步（pytest），未执行 1e6/1e7 步。
- [x] **未做多级协同**
  - 仅单机任务级；无多机/僚机/HVA。
- [x] **未补完整 2.6/2.7 战术树**
  - 现有 `引导策略行为树/llm_bt` schema 未改动；新增独立 py_trees 调度桥。
- [x] **未实现完整 `jsrl_trainer.py`/课程衰减**
  - 仅 `guide_action/guide_steps` 透传占位，并在文档中标注。
- [x] **未破坏现有接口与 CLI**
  - `接口规范.md` v1.0 的 Dict 动作/帧/info/错误码未改；仅补充 `control_mode` 兼容说明；`agents/__init__.py` 仅新增导出；当前三套全量 187 项测试通过。
- [x] **未提交 API Key**
  - 本轮未调用任何付费 LLM API；无密钥写入。

---

## 2. 交付物核对

- [x] 工作区代码改动（两个本地 commit，未 push）
- [x] `docs/阶段三_专家智能体DRL_实现说明.md`
- [x] `docs/阶段三_专家智能体DRL_测试报告.md`
- [x] `docs/阶段三_专家智能体DRL_验收清单.md`
- [x] `docs/阶段三_专家智能体DRL_命令清单.md`
- [x] `docs/阶段三_稳定飞行基线_实现与测试报告.md`
- [x] 最小演示产物：`docs/assets/stage3_expert_demo/`
- [x] 稳定飞行产物：`多任务智能体/validation/results/stage3_stability/`
- [x] 最终回复：≤300 字中文摘要

---

## 3. 明确未完成 / 后续项（不构成本轮失败）

| 项 | 状态 | 后续动作 |
|---|---|---|
| 完整 JSRL 训练循环（课程衰减、引导样本、内在奖励） | 未实现 | 阶段三 3.1：新增 `jsrl_trainer.py`，接 `get_action(guide_action, guide_steps)` |
| 混合动作 PPO（连续 4 + Bernoulli 3，掩码进分布） | 🟡 网络实现+2048步冒烟通过；命中率/长训练未验收 | 后续：固定测试集长训练，目标 >60% 命中率 |
| LSTM / frame_stack 部分可观测 | 未实现 | 阶段三 P2 |
| 多机巡逻/防御/HVA/多方向来袭 | 未实现 | `common/bvr_multi_env.py` 扩展，保持 ActionV1 不变 |
| 80~120 km / 600 步场景参数化、`obs_v2` 4 维任务标签 | 未实现 | 阶段三 3.2/3.3 |
| 任务性能（命中率/拦截率/HVA 存活率/EW 探测概率） | 未验证 | 完整训练 + 固定测试集评估 |
| 现有 LLM 行为树 schema 接入专家叶节点 | 未接入 | 阶段三/四把 `ExpertAgentControl` 并入动作目录 |
| 模型入库 | 未入库（`.gitignore`） | 正式模型另行归档/发布 |

---

## 4. 验收结论

本轮满足任务书底线：

> 4 个任务级智能体可实例化、可小规模训练、可被行为树按任务名调用、输出合法 ActionV1；assisted 240 步隔离飞行 24/24 不坠机；5 个技能智能体不退化；相关 pytest 全绿。

未声称完成完整 JSRL、多机协同、完整战术树或任务性能达标。
