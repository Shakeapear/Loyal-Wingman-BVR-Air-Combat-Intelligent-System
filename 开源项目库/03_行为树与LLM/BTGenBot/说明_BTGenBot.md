# BTGenBot（LLM 生成行为树，IROS 2024）

## 简介

BTGenBot（IEEE IROS 2024，arXiv:2403.12761）演示了用**轻量 LLM（≤7B：llama-2-7b-chat、codellama-7b）**微调后生成机器人行为树（XML）的完整管线：指令数据集构建（GPT-3.5 蒸馏）→ LoRA 微调 → 语法/语义验证 → 仿真/实物执行。与本项目"LLM 生成空战行为树 + 验证"模块**同构**，是目前最贴合的公开参考实现。

- 仓库：https://github.com/AIRLab-POLIMI/BTGenBot
- 许可证：MIT
- 论文数据集：https://huggingface.co/datasets/AIRLab-POLIMI/btgenbot
- LoRA 适配器：HF 上 AIRLab-POLIMI 同名空间（llama-2-7b-chat-hf-btgenbot-adapter 等）
- 源码位置：`源码\bt_generator\`（生成+微调）、`源码\bt_validator\`（验证器）、`源码\dataset\`（微调数据集）、`源码\prompt\`（zero/one-shot 提示词）

## 选择理由

1. 计划书阶段二的 LLM 行为树生成（动作库 XML 节点 + 提示词 + 验证器）在本项目中无现成代码，此仓库提供了**完整可复用的流程模板**；
2. 验证器（bt_validator）与本项目"LLM 输出语法/语义校验"需求直接对应；
3. 支持 7B 级本地模型微调（LoRA），为项目在无 GPT-4o API 条件下提供了本地化方案。

## 运行说明

```powershell
# 1. 环境（bt_generator）
conda create -n btgenbot python=3.10
pip install -r 源码\bt_generator\requirements.txt

# 2. 用微调模型生成行为树（GUI 或 notebook）
python 源码\bt_generator\btgenbot.py        # GUI：输入任务描述→输出行为树 XML
# 或打开 源码\bt_generator\inference.ipynb（需下载 HF 上的 LoRA 适配器）

# 3. 验证（bt_validator 基于 ROS2+BehaviorTree.CPP，Linux 环境）
colcon build && ros2 launch bt_validator ...
```

> 组员学习路径：`源码\prompt\`（提示词模板，**可直接借鉴改造为中文空战提示词**）→ `源码\dataset\`（指令-行为树对的数据格式）→ `bt_validator`（校验逻辑）。
> 注：bt_client/bt_validator 依赖 ROS2（Linux）；仅参考生成与数据部分时，Windows 上只需 bt_generator。
