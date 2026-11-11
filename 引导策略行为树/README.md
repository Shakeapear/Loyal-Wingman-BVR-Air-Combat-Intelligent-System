# 引导策略行为树（步骤 2.5–2.7 子系统）

**定位：** 本目录承载《大创项目实施计划书》阶段二步骤 2.5–2.7 的**引导策略行为树**子系统——
LLM 行为树生成器-测试器框架（2.5，已交付）、二级优先级引导策略子树（2.6，已交付）、
行为树仿真验证（2.7，已交付）、教师树 v3 与 1v1 对抗评估（阶段三）。子系统与
`多任务智能体/` 为**单向依赖**：只调用环境数据/动作接口，不含 DRL 训练代码。

> 当前状态（2026-11-11）：2.5–2.7 均已交付；阶段三教师基线固定为
> `teacher_v3_speed260`（修复后 solo default/bvr 均 96/86，误发射 0、
> 安全违规 0、trace 50/50；对握手写树 combat 仍落后）。本目录测试 90 项，
> 三套全量 227 项通过。

## 一、步骤 2.5 交付物索引

| 交付物 | 内容 | 位置 |
|--------|------|------|
| D2.5-1 | Coder-Executor-Tester 框架源码 | `llm_bt_generator.py` + `llm_bt/` 包 |
| D2.5-2 | Prompt 模板库（Coder/Tester 各 1 份，每份 ≥1KB） | `prompts/coder_prompt.md`、`prompts/tester_prompt.md` |
| D2.5-3 | 迭代收敛日志样例（生成→测试→修正完整循环） | 每次运行自动生成 `outputs/<时间戳>/iteration_log.md`；归档样例：`examples/iteration_log_离线两轮样例.md`（含 bt_final.json / 文本树样例） |

## 二、框架结构（Coder → Executor → Tester 闭环）

```
Prompt(prompts/*.md)                                        反馈(缺失节点/失败用例)
      │                                                             ▲
      ▼                                                             │
┌──────────┐  行为树JSON   ┌──────────────┐   py_trees 树   ┌──────────────┐
│  Coder   │ ───────────▶ │ 静态校验      │ ─────────────▶ │   Executor   │
│ (LLM/mock)│              │ JSON schema + │                │ JSON→py_trees │
└──────────┘              │ bt_validator  │                │ →JSBSim 执行  │
                          └──────────────┘                └──────┬───────┘
                                                                 ▼
                                                         ┌──────────────┐
                                                         │   Tester     │
                                                         │ 5 组用例+评估 │
                                                         └──────────────┘
```

- **Coder**（`llm_bt/llm_client.py` + prompts）：DeepSeek（默认）/OpenAI/离线 mock，
  输出 nodes+edges 行为树 JSON（规范见 `llm_bt/schema.py`）；
- **静态校验**（`llm_bt/validator_bridge.py`）：JSON 规范校验 + 转换为 BehaviorTree.CPP
  XML 后复用 BTGenBot 纯 Python `bt_validator`（节点库为 BVR 动作/条件目录）；
- **Executor**（`llm_bt/executor.py`）：JSON → py_trees（反应式组合节点）→ BVRCombatEnv
  1 Hz 决策循环；动作经实测整定的指令回路（坡度内环/俯仰符号/油门微调）映射为
  Dict 动作；全程记录节点执行日志；
- **Tester**（`llm_bt/tester.py`）：LLM 生成 5 组测试用例（种子化初始态势 + 判据），
  逐组在环境中执行并统计通过率、安全违规、OODA 覆盖与交战结果。

**收敛判据**（计划书步骤 2.5）：用例通过率 > 90%、OODA 关键节点覆盖 > 95%、安全违规 = 0。

### 统一引导契约（2026-11-03）

行为树不再直接向训练循环暴露 Dict 动作，而是经 `BTGuideProvider` 输出
`GuideStep`：

```python
GuideStep(action, active, source, guide_step, control_mode, mask, meta)
```

- `action`：ActionV1 7 维；未激活时为 `None`；
- `control_mode`：`assisted` 或 `raw_fcs`，必须与环境 `control_mode` 一致；
- `source`：`bt` / `human` / `drl` / `fallback`；
- `guide_step`：本 episode 已引导步数；
- `mask`：ActionMaskV1（可选，由调用方注入）。

`BTGuideProvider` 负责 `GuideInterface` 步数、`BoundaryConstraints` 守卫、
行为树 tick 和 `GuideStep` 封装。`assisted` 模式下行为树的
`flight` 前 3 维解释为速度/坡度/高度设定值，不再是 raw FCS 舵面。

招式库通过 `llm_bt.maneuvers.ManeuverRegistry` 读取
`飞行数据集资料库/05_人类战术动作与招式库/04_标注与招式表/maneuver_catalog_v1.csv`；
可用 `BVR_MANEUVER_CATALOG` 覆盖路径。招式只作为注册表条目，不增加
ActionV1 维度；行为树通过单一动作调用：

```json
{"type": "Action", "name": "execute_maneuver",
 "args": {"maneuver": "crank", "parameters": {"offset_deg": 60}}}
```

`execute_maneuver` 由 `maneuver_actions.py` 的 15 个控制器执行；静态校验
在传入 `maneuver_ids` 时会拒绝未注册招式。

80–120 km 参考树：

```bash
python -X utf8 run_2_7_validation.py \
  --tree handwritten_80_120 --scene bvr80-120 \
  --seeds 1-20 --no-acmi --out-dir outputs/validation_2_7/handwritten_80_120
```

2026-11-03 结果：存活 80%、命中 75%、误发射 0%、安全 0、OODA 执行 97%、
node_trace 20/20；30–80 km 回归 90%/75%。

LLM 生成器同样支持场景：

```bash
python -X utf8 llm_bt_generator.py --provider mock --scene bvr80-120 --rounds 3
# 在线最终树：先在同一 PowerShell 会话设置 DEEPSEEK_API_KEY，再
python -X utf8 llm_bt_generator.py --provider deepseek --scene bvr80-120 --rounds 3
```

mock 只验证生成/校验/测试流程，不是战术教师；当前 mock 树 20 seeds 复核为
存活 60%/命中 5%。DeepSeek 在线候选已用仓库外密钥生成并复核。旧 `deepseek-chat` 候选约
存活 10%/命中 5%，被拒绝；改用 `deepseek-flash`（DeepSeek V4.1 Flash 1017）
后，候选树外部 20 seeds 达到 **存活 80%/命中 75%/误发射 0%/安全 0/OODA 97%/
trace 20/20**。

进一步用 `tools/search_bt_candidates.py` 做外部评分闭环搜索，g1c2 候选
50 seeds 达到 **存活 92%/命中 80%/误发射 0%/安全 0/trace 50/50**，已归档到
`docs/assets/3.5_llm_tree_search/`。该树在 30–80 km 只有 60%/60%，所以采用
场景路由：

```text
default      → 手写树（90%/75%）
bvr80-120    → LLM g1c2 树（92%/80%）
```

清单：`tree_manifest.json`；加载：`llm_bt.tree_registry.resolve_tree(scene)`。

**OODA 关键节点**（`llm_bt/schema.py:OODA_REQUIRED`）：观察 `search_target`/`has_target_detected`、
判断 `lock_target`/`is_in_launch_zone`、决策 `has_weapon_remaining`/`is_fuel_low`、
行动 `fire_missile`/`evade_missile`（共 8 项，全含为 100%）。

## 三、快速开始（DC 环境，在 `引导策略行为树/` 目录下）

```powershell
# 1) 离线确定性验证（无需 API Key；含 生成→测试→修正 两轮完整日志）
python -X utf8 llm_bt_generator.py --provider mock --rounds 3

# 2) 真实 LLM（DeepSeek；需先设置 DEEPSEEK_API_KEY）
python -X utf8 llm_bt_generator.py --provider deepseek --rounds 3

# 3) OpenAI 兼容端点 / 自定义任务 / 快速冒烟（跳过攻击区解算）
$env:OPENAI_API_KEY = "sk-..."
python -X utf8 llm_bt_generator.py --provider openai --model gpt-4o
python -X utf8 llm_bt_generator.py --mission "巡逻并拦截入侵目标" --fast

# 4) 单元测试（19 用例）
python -m pytest tests/test_llm_bt.py -v
```

产物（每轮 + 汇总）写入 `outputs/<运行时间戳>/`：
`bt_vN.json/.xml`、`tree_vN.txt/.dot`、`test_cases_vN.json`、
`test_report_vN.json`（含每用例指标）、`llm_*_raw_vN.txt`（LLM 原始输出）、
`bt_final.json/.xml`、`iteration_log.md`（D2.5-3 迭代收敛日志）。

**离线运行预期**：Round 1 生成仅攻击的简化树（OODA 覆盖 75%，缺
`evade_missile`/`is_fuel_low`）→ 未收敛；反馈携带缺失节点，Round 2 补齐
Self_Defense / Fuel_Return 分支后覆盖 100%、通过率 100%、安全违规 0 → 收敛。

**环境变量**：`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`（真实 LLM）；
`DEEPSEEK_BASE_URL`、`OPENAI_BASE_URL` 可指向代理/兼容服务。

## 四、行为树可视化

- `tree_vN.txt`：终端可读文本树形图（含动作参数）；
- `tree_vN.dot`：graphviz 源文件，可渲染为图片：
  ```powershell
  dot -Tpng outputs/<run>/tree_v2.dot -o tree_v2.png
  ```
- `bt_vN.xml`：BehaviorTree.CPP 兼容格式（Selector→Fallback），可进 Groot 等工具。

## 五、接入仿真与录制（人工复盘 / 接口联调）

`run_tree_sim.py` 把行为树接入 BVRCombatEnv 跑完整一局双机对抗并录制全量数据：

```powershell
python -X utf8 run_tree_sim.py --seed 9                # 单局：CSV + ACMI + summary
python -X utf8 run_tree_sim.py --seeds 1,2,3 --quiet   # 多种子扫描
```

产物在 `outputs/sim/`：`bt_ep_seed<N>.csv`（逐帧全量数据）、
`bt_ep_seed<N>.acmi`（双击进 TacView 3D 复盘）、`bt_ep_seed<N>.summary.json`。

**2026-09-17 联调结果**（24 种子扫描，默认 240 s/局）：
- 8 局跑满 240 s 完整对抗。样例 seed 9：双方各发射 2 弹、MAWS 告警 36 s、
  最近距 4.2 km、无命中超时——全程 241 帧无缺失，ACMI 含 2 机 × 241 帧、
  4 枚导弹对象、253 行锁定连线、射击日志 4 条；
- 其余多为我方快速命中获胜（34~86 s），1 局 129 s 被命中；
- 全部 24 局：0 逻辑错误、0 安全违规。

**联调中发现并修复的问题**：
1. 发射后 F-pole 偏转指令被默认巡逻分支在同一决策步覆盖 → 生成树改为
   「交战失败回退 `lock_target` 继续跟踪」结构（见 `llm_bt/mock_llm.py` 与
   Coder Prompt 规则 7）；
2. ECM 全程未开（敌首发过早）→ 搜索/锁定/规避动作确保开启 ECM；
3. 发射后不做偏转机动（迎面吃弹）→ `lock_target` 在自弹在飞期间执行
   crank 60° 偏转（`executor.py`）。

**数据链路**：executor（Dict 动作）→ BVRCombatEnv（`get_viz_frame()` 帧）→
可视化工具 `EpisodeRecorder`（CSV）→ `acmi.export_acmi`（TacView）——
接口与数据格式已按 2.4 工具链统一适配。

## 六、测试用例指标（`llm_bt/tester.py`，Tester Prompt 中声明）

| 指标 | 含义 |
|------|------|
| no_logic_error | 执行期间无非法动作/异常（1=无） |
| safety_violations | 安全包线违规步数（过载/马赫/高度/速度越界） |
| no_crash | 未坠毁（1=未坠毁） |
| steps_executed | 实际决策步数 |
| fired_own / in_zone_steps / enemy_fired | 交战过程量 |
| enemy_hit / own_not_hit | 命中/生存结果 |
| engaged | 进入交战（发射/进攻击区/敌方发射，1=是） |

离线 mock 默认用例的通过判据为**安全/逻辑/有效飞行**（no_logic_error、safety_violations=0、
no_crash、steps≥30）；任务效果类判据（生存、命中、交战）由真实 LLM Tester 按
Prompt 生成或人工追加——离线默认判据不构成对树战术优劣的最终结论，交战/生存
数据仍在每轮报告中完整记录。

## 七、限制与后续（2.6 / 2.7）

- **单机 1v1 环境限制**：`maintain_formation` 无僚机对象（接口预留，按平飞处理）；
  `has_human_command` 默认无外部输入（黑板可置位）；雷达/锁定由环境自动状态机实现，
  `search_target`/`lock_target` 表现为航向/跟踪指令；
- **2.6 已实现（2026-11-01，最小可运行版）**：
  - `llm_bt/boundary_constraints.py`：第一优先级边界约束守卫（低高度/最小速度/
    过载/最大速度/禁飞区/低燃油，含滞回、覆盖日志与过载操纵限幅）；
  - `llm_bt/tactical_execution_tree.py`：第二优先级战术执行子树（OODA 8/8，
    Human_Command / Self_Defense / Fuel_Return / BVR_Engagement / Default_Patrol）；
  - `llm_bt/guide_interface.py`：JSRL 引导接口（`get_action` /
    `update_guide_steps` / `reset`，耗尽返回 `(None, False)`）；
  - `run_guidance_sim.py`：3 seeds × 240 步联调、CSV/ACMI/summary/约束日志；
  - 结果：OODA 执行覆盖 100%、0 安全违规、0 逻辑错误；行为树 47 项、三套全量
    150 项测试通过。报告：`docs/阶段二_2.6_引导策略行为树_实现与测试报告.md`，
    命令：`docs/阶段二_2.6_命令清单.md`；
- **2.7 已实现（2026-11-01）**：`run_2_7_validation.py` 对 20 组固定态势分别执行
  手写树与 2.5 在线 LLM 树（同一守卫/引导/240 步），输出 CSV/ACMI/summary/
  events/node_trace 与对比图表；结果：两树存活率 90%（18/20）、命中率 75%、
  误发射率 0%、安全违规 0、trace 20/20；报告见
  `docs/阶段二_2.7_行为树仿真验证报告.md`，阶段二总结见
  `docs/阶段二_技术总结报告.md`。

## 八、教师树 v3 与 1v1 对抗评估（2026-11-11 中期基线）

- 树注册：`tree_manifest.json` + `llm_bt/tree_registry.py`；场景
  `teacher_v3_speed260_default` / `teacher_v3_speed260_bvr80-120` 指向
  `docs/assets/3.7_teacher_v3_speed260/bt_v3_speed260.json`；
- 构建：`llm_bt/tactical_execution_tree.py` 的 `tactical_tree_v3_json(profile)`，
  profile 数据驱动（首发/补射动作、防御招式、搜索速度、crank 偏角）；
- 对抗入口：`run_bt_vs_bt.py`（双树 1v1、对称燃油/逃逸终止、逐发记录）；
  可视化：`visualize_bt_vs_bt.py`；
- 修复后 50 seeds 指标：solo default/bvr 均 96/86；h2h vs 手写 default
  26-40-34（combat Pk 17.3%，combat wins 26 < 手写 40，生存 60%）、bvr80-120
  40-31-29（combat 11 < 手写 29，Pk 10.6%，29 场对手逃逸胜，生存 71%）。
  评估必须区分 `combat_wins` 与 `non_combat_wins`；当前教师树尚未在 combat
  上超过手写树，需在中期前重新整定高度比例与防御招式；
- 入口共享工具：`llm_bt/sim_common.py` 统一种子解析、EpisodeRecorder、边界守卫与
  ACMI 导出；各入口 CLI 和产物格式不变。测试路径由各 `tests/conftest.py` 统一引导；
- 复现：

```powershell
$py = "C:\Users\36266\.conda\envs\DC\python.exe"
& $py -X utf8 run_bt_vs_bt.py --scene bvr80-120 --seeds 1-50 --record-launches
& $py -X utf8 run_2_7_validation.py --tree handwritten --scene bvr80-120 `
  --seeds 1-50 --no-acmi
```

## 九、依赖

- Python：py_trees 2.5.0（DC 环境已装）、numpy；环境侧 `多任务智能体/common`；
- 复用：`开源项目库/03_行为树与LLM/BTGenBot/Python重写_bt_validator/bt_validator.py`
  （静态校验，按文件路径导入）；
- 网络仅真实 LLM 模式需要（DeepSeek/OpenAI 兼容端点）。
