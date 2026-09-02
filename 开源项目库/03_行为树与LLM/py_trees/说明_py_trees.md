# py_trees（Python 行为树框架）

## 简介

py_trees 是 Python 生态中行为树的**事实标准**框架（BSD 许可，持续活跃更新），提供完整的行为树节点体系（Sequence/Selector/Parallel/Decorator）、可视化（py_trees_ros_viewer/dot 图）与黑板（blackboard）机制。BVRGym 的行为树对手、众多机器人项目均基于它。

- 仓库：https://github.com/splintered-reality/py_trees
- 许可证：BSD（源码包内含 LICENSE）
- 版本：2.5.0（`安装包\py_trees-2.5.0-py2.py3-none-any.whl`）
- 源码位置：`源码\`（自 wheel 解包：核心 + demos + parsers）

## 选择理由

1. 计划书阶段二"LLM 生成行为树"的执行层需要 Python 行为树运行时，py_trees 是最成熟选择；
2. 与 BVRGym 参考行为树（`jsb_gym/bts`）同一框架，节点写法可直接对照；
3. 内置 `py_trees.display` 可视化，便于组内评审 LLM 生成的树结构。

## 运行说明

```powershell
pip install py_trees   # 官方 PyPI 包（2.5.0）
# 或离线安装本目录包：pip install "开源项目库\03_行为树与LLM\py_trees\安装包\py_trees-2.5.0-py2.py3-none-any.whl"
```

最小示例（空战机动节点雏形）：

```python
import py_trees

class 敌机已锁定(py_trees.behaviour.Behaviour):    # 条件节点
    def update(self):
        return py_trees.common.Status.SUCCESS if self.agent.locked else py_trees.common.Status.FAILURE

class 机动规避(py_trees.behaviour.Behaviour):      # 动作节点
    def update(self):
        self.agent.cmd = ("滚转", -0.5)            # 输出机动指令
        return py_trees.common.Status.RUNNING

root = py_trees.composites.Selector("顶层策略", memory=True)
root.add_children([py_trees.composites.Sequence("威胁优先", [敌机已锁定("锁定判断"), 机动规避("规避")]),
                   其他分支...])
root.tick_once()
py_trees.display.ascii_tree(root)                  # 打印树结构
```

> 组员学习路径：`源码\demos\`（官方示例）→ `py_trees\behaviour.py`（节点基类）→ `py_trees\composites.py`（组合节点）。
