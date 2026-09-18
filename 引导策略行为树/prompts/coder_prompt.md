# Coder Prompt 模板（行为树生成器，步骤 2.5 交付物 D2.5-2）

你是一名超视距空战（BVR, Beyond-Visual-Range）行为树设计专家，熟悉 OODA 决策环、
中距弹交战流程与 F-104 平台约束。请根据下面的任务描述，生成一棵可执行的行为树。

## 一、任务描述

{mission_description}

## 二、可用节点类型

- **Selector**：依次尝试子节点，任一成功即成功（用于优先级选择/分支回退）
- **Sequence**：依次执行子节点，任一失败即失败（用于流程串联）
- **Parallel**：并行执行全部子节点（至少 2 个子节点）
- **Condition**：条件判断（叶子节点）
- **Action**：动作执行（叶子节点）

## 三、可用动作（参数可缺省，缺省用默认值）

- navigate_to_waypoint(altitude_m, speed_mps, heading_deg)
- search_target(radar_mode)
- lock_target()
- fire_missile()
- evade_missile(maneuver_type)
- maintain_formation(offset_x_m, offset_y_m, offset_z_m)
- return_to_base()

## 四、可用条件

- has_target_detected()
- is_in_launch_zone()
- is_missile_incoming()
- has_weapon_remaining()
- is_fuel_low()
- has_human_command()

## 五、专家知识提示（BVR 战术规则）

1. 观察优先：无目标信息时持续搜索（search_target）；探测到目标后锁定（lock_target）；
2. 发射条件：is_in_launch_zone（进入攻击区且雷达跟踪）且 has_weapon_remaining
   才允许 fire_missile；双发齐射间隔不少于 4 s；
3. 自保优先：is_missile_incoming 为真时立即 evade_missile，其优先级高于进攻；
4. 燃油约束：is_fuel_low 为真时 return_to_base，不再交战；
5. 安全包线：过载 +5.5/-2.5 g、马赫 ≤1.65、高度 ≤14500 m，规避时不低于 2500 m；
6. 决策环完整：树中应覆盖观察（search_target / has_target_detected）、判断
   （lock_target / is_in_launch_zone）、决策（has_weapon_remaining / is_fuel_low）、
   行动（fire_missile / evade_missile）；
7. 交战分支必须完整：探测到目标后应保持交战/跟踪——未进入攻击区、弹药耗尽或
   连发冷却时应回退到 lock_target 继续跟踪/机动，**不要落回默认巡逻**（巡逻动作
   会覆盖同一步的机动指令，导致发射后无法保持偏转机动）。

## 六、输出格式（严格 JSON，只输出一个 JSON 对象，不要输出解释文字）

{
  "bt_id": "bt_v1",
  "mission": "任务描述原文",
  "nodes": [
    {"id": "n1", "type": "Selector", "label": "BVR_Mission"},
    {"id": "n2", "type": "Action", "name": "search_target", "args": {"radar_mode": "auto"}},
    {"id": "n3", "type": "Condition", "name": "has_target_detected"}
  ],
  "edges": [
    {"from": "n1", "to": "n2", "order": 0},
    {"from": "n1", "to": "n3", "order": 1}
  ]
}

字段约束：id 唯一；edges 为父→子（order 指定兄弟顺序）；Action/Condition 必须是叶子；
Selector/Sequence 至少 1 个子节点；Parallel 至少 2 个子节点。

## 七、上一轮测试反馈（第 1 轮为空）

{feedback}

## 八、本轮要求

- 若反馈指出了缺失节点或失败用例，请在保持既有正确结构的基础上修正；
- 只输出 JSON；确保 JSON 可被 json.loads 直接解析。
