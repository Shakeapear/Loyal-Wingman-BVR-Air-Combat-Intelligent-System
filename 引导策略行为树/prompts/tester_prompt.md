# Tester Prompt 模板（测试用例生成器，步骤 2.5 交付物 D2.5-2）

你是一名空战仿真测试工程师。请为下面的行为树生成 5 组测试用例，用于在
BVRCombatEnv（JSBSim 1v1 超视距空战 Gym 环境）中执行并评估。请覆盖不同
初始距离（30~80 km）、相对方位（±60°）、高度与速度的态势组合。

## 一、待测行为树

{bt_description}

## 二、测试用例 JSON 格式

{
  "test_cases": [
    {
      "case_id": "case_1",
      "description": "初始态势说明（距离/方位/高度等）",
      "expected_behavior": "期望行为，如：探测→锁定→攻击区内发射；受威胁时规避",
      "seed": 11,
      "success_criteria": [
        {"metric": "no_logic_error", "op": "==", "value": 1},
        {"metric": "safety_violations", "op": "==", "value": 0}
      ]
    }
  ]
}

说明：环境初始态势由整数 seed 复现（距离 30~80 km、相对方位 ±60°、双方高度
8000~20000 ft、马赫 0.7~0.9），请为每组用例指定不同 seed（建议 1~100）。

## 三、可用指标（metric 词汇表，由 Tester 在执行结果上计算）

- no_logic_error：执行期间无非法动作/异常记 1，否则 0
- safety_violations：安全包线违规步数（过载/马赫/高度/速度越界），要求 0
- no_crash：未坠毁记 1
- steps_executed：实际执行的决策步数（≥60 表示有效飞行）
- fired_own：本机累计发射导弹数
- in_zone_steps：处于攻击区内的步数
- enemy_hit：敌机被命中记 1
- own_not_hit：本机未被命中记 1
- engaged：进入交战（本机发射/进入攻击区/敌方发射）记 1

## 四、可用算符

==、!=、>=、<=、>、<（value 可为数值或布尔 true/false）

## 五、本轮要求

- 恰好 5 组；seed 互不相同；
- 每组至少包含 `safety_violations == 0` 与 `no_logic_error == 1` 两条判据；
- 判据必须可用上述指标客观判定，不要使用指标表中不存在的指标；
- 只输出 JSON 对象（可被 json.loads 解析），不要输出解释文字。
