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
- execute_maneuver(maneuver, parameters)  # 参数化人类招式；maneuver 必须来自下方招式库

### 可用招式库（`execute_maneuver.maneuver`）

`search_pattern, intercept_closure, lock_and_track, crank, notch, break_turn,
barrel_roll, defensive_spiral, launch_and_crank, extend_and_reengage, split_s,
loop, immelmann, fuel_rtb, ecm_on`

只能使用上述 ID；`parameters` 为对象，可传 `offset_deg`、`speed_mps`、
`duration_s` 等参数。不要发明招式名。

## 四、可用条件

- has_target_detected()
- is_in_launch_zone()
- is_in_nez()
- is_missile_incoming()
- has_weapon_remaining()
- is_fuel_low()
- has_human_command()
- is_target_hot()          # 目标 aspect < 60°，迎面/热姿态
- is_target_turning()      # 目标航向变化率 > 1°/s（防御性转弯）
- is_shot_window_open()    # 无 MAWS 警报 + 目标未转弯（干净补射窗口）
- is_closure_high()        # 闭合率 > 300 m/s
- is_opening()             # 目标正在拉开
- is_within_dynamic_lar()  # 动态发射包线（热姿态 + 高闭合 + Rmax）
- is_within_effective_range()  # 攻击区内且距离 <= 有效发射距离上限
- is_near_boundary()       # 距基地 > 80% 逃逸半径
- has_missile_in_flight()  # 本机导弹仍在飞
- has_full_weapons()       # 本机仍有 2 发（Rmax 首发射击窗口）
- is_rwr_locked()          # 被敌雷达锁定
- is_tta_long()            # MAWS TTA > 8 s

## 五、专家知识提示（BVR 战术规则）

1. 观察优先：无目标信息时 `search_target` + `execute_maneuver(search_pattern)`；
   探测到目标后 `lock_target` + `execute_maneuver(intercept_closure)` 闭合接敌；
2. 发射条件（先手优先）：第一发用 `is_in_launch_zone` + `is_within_dynamic_lar`
   + `is_target_hot` + `has_weapon_remaining` 允许 `fire_missile`，在 Rmax 附近
   热姿态先手；`is_in_nez` 只作为第二发/近距离补射条件，不要等 NEZ 才打第一发。
   目标 beam/cold（`is_target_hot` 为假）或 `is_opening` 为真时不要发射，继续闭合。
   双发齐射间隔不少于 4 s；**不要在同一个 Sequence 里先 `fire_missile` 再
   `execute_maneuver(launch_and_crank)`**（第二次发射会因冷却失败并中断分支）；
   要么只用 `launch_and_crank` 作为发射动作，要么 `fire_missile` 后回退到
   `lock_target` + `execute_maneuver(crank)`；
3. 自保优先：`is_missile_incoming` 为真时立即 `evade_missile`，其优先级高于进攻；
   远距/长 TTA（`is_tta_long`）先用 `notch`，近距/短 TTA 用 `break_turn`；
   不要在 80–120 km 场景默认使用 `notch`（实测生存率显著下降）；
4. 燃油约束：is_fuel_low 为真时 return_to_base，不再交战；
5. 安全包线：过载 +5.5/-2.5 g、马赫 ≤1.65、高度 ≤14500 m，规避时不低于 2500 m；
6. 决策环完整：树中应覆盖观察（search_target / has_target_detected）、判断
   （lock_target / is_in_launch_zone）、决策（has_weapon_remaining / is_fuel_low）、
   行动（fire_missile / evade_missile）；
7. 交战分支必须完整：探测到目标后应保持交战/跟踪——未进入攻击区、弹药耗尽或
   连发冷却时应回退到 lock_target 继续跟踪/机动，**不要落回默认巡逻**（巡逻动作
   会覆盖同一步的机动指令，导致发射后无法保持偏转机动）。发射后应优先进入
   `has_missile_in_flight` 分支，用 `lock_target` + `execute_maneuver(crank)`
   保持偏转，不要立即回到搜索。
8. 80–120 km 场景必须包含远距搜索/闭合：`execute_maneuver(search_pattern)`、
   `execute_maneuver(intercept_closure)`；只调用 `search_target` 平飞会在
   远距场景中逃逸。搜索分支要有边界意识：`is_near_boundary` 为真时先
   `execute_maneuver(fuel_rtb)` / `return_to_base`，不要继续向外扫。

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

