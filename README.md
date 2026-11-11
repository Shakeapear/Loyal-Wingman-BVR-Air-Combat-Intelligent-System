# 忠诚僚机 BVR 空战智能系统

面向忠诚僚机超视距空战的知识/数据双驱动研究项目。仓库包含自研仿真环境、任务级 DRL 智能体、
引导策略行为树、可视化工具，以及阶段交付文档与实验证据。

## 目录地图

| 目录 | 内容 | 入口 |
|---|---|---|
| `多任务智能体/` | JSBSim 环境、导弹/雷达模型、技能与任务智能体、PPO/混合动作训练 | `多任务智能体/README.md` |
| `引导策略行为树/` | LLM 行为树生成器、行为树执行器、边界约束、招式库、v3 教师树、1v1 对抗评估 | `引导策略行为树/README.md` |
| `可视化工具/` | 2D 战术显示、CSV/GIF/MP4 回放、TacView ACMI 导出 | `可视化工具/README.md` |
| `docs/` | 阶段二/三报告、命令清单、验收材料与实验证据 | `docs/README.md` |
| `JSBSim/` | JSBSim 运行时与 F-104 定制数据 | `JSBSim_培训与验证资料/README.md` |
| `资料/`、`开源项目库/`、`飞行数据集资料库/` | 文献、参考源码与飞行数据（多数大文件按 `.gitignore` 不入库） | 各目录说明 |

## 运行环境

本地开发使用 conda 环境 `DC`（Python 3.12，CPU 版 torch / gymnasium / stable-baselines3 /
py_trees / jsbsim）。Windows 下默认解释器：

```powershell
C:\Users\36266\.conda\envs\DC\python.exe
```

依赖版本以 `多任务智能体/requirements-server.txt` 为准；服务器安装与预检见该文件头部说明和
`多任务智能体/validation/preflight_server.py`。密钥只从环境变量 `DEEPSEEK_API_KEY` 或仓库外
密钥文件读取，禁止写入仓库（见 `AGENTS.md`）。

## 快速验证

```powershell
$py = "C:\Users\36266\.conda\envs\DC\python.exe"

# 全量测试（227 项）
& $py -X utf8 -m pytest "多任务智能体/tests" "引导策略行为树/tests" "可视化工具/tests" -q

# 行为树 1v1 对抗（固定 seeds）
& $py -X utf8 "引导策略行为树/run_bt_vs_bt.py" `
  --scene default --seeds 1-2 --record-launches

# 2.7 行为树验证（固定 seeds）
& $py -X utf8 "引导策略行为树/run_2_7_validation.py" `
  --tree handwritten --scene default --seeds 1-2 --no-acmi
```

## 当前基线（2026-11-11）

- 教师树：`teacher_v3_speed260`（solo default 100/86、bvr80-120 98/86；误发射 0、安全违规 0、trace 50/50）。
- 对抗：default 19-21-60（combat Pk 12.4%）；bvr80-120 42-14-44（combat Pk 10.7%，其中 31 场为对手逃逸胜）。
- 场景路由与指标口径见 `引导策略行为树/tree_manifest.json`、`项目状态与进度总览.md`。
- 回退点：`checkpoint-midterm-20261111`。

## 文档优先级

1. `项目状态与进度总览.md`：项目全局状态唯一权威来源；
2. `接口规范.md` / `接口文档.md`：数据契约与逐接口参考；
3. `docs/`：阶段报告、命令清单与验收材料；
4. 各模块 README：模块内使用说明。

计划细节见 `大创项目实施计划书.md` 与 `研究计划与实施方案_详细版.md`；与状态总览冲突时以状态总览为准。
