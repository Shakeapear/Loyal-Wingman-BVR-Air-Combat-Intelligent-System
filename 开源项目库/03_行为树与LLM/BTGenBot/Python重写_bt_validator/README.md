# Python 重写：BTGenBot bt_validator

**原版**：`../源码/bt_validator/`（C++，ROS2 + BehaviorTree.CPP，执行行为树并在机器人仿真中验证）
**重写**：`bt_validator.py`（纯 Python 静态校验器，保留原版验证逻辑的静态部分）

## 重写对照

| 原版（C++）逻辑 | Python 重写实现 |
|----------------|----------------|
| XML 解析（BTCPP 格式） | `xml.etree.ElementTree` 解析 `<root>`/`<BehaviorTree>` |
| 节点类型注册检查（factory.registerNodeType） | `DEFAULT_CATALOG` 节点库 + 可外置 JSON（`--catalog`） |
| 树结构合法性（单主树、组合节点子节点数） | `CONTROL_RULES` 约束检查 |
| 仿真执行验证（机器人） | 预留 evaluator 接口，动态验证由本项目 JSBSim 环境承担 |

## 用法（DC 环境）

```powershell
# 校验单个文件
& "C:\Users\36266\.conda\envs\DC\python.exe" bt_validator.py 测试树样例\tree1.xml

# 校验整个目录（原版全部 10 个测试树 + demo）
& "C:\Users\36266\.conda\envs\DC\python.exe" bt_validator.py 测试树样例
```

输出示例：`校验通过: 0 错误, 2 警告`（未知节点类型会提示注册节点库）。
