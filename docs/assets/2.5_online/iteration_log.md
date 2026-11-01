# 迭代收敛日志（D2.5-3）

- 时间：2026-11-01 09:34:29
- provider：deepseek（model=deepseek-chat）
- 任务：超视距空战拦截：雷达搜索并锁定目标，进入攻击区后发射中距弹（双发间隔≥4 s），导弹来袭时优先规避，燃油不足时返航，全程满足安全包线约束
- 契约版本：{'frame': 'frame_v1', 'csv': 'csv_v1', 'info': 'info_v1', 'result_record': 'record_v1', 'test_case': 'test_case_v1', 'bt_schema': 'bt_v1'}
- 收敛判据：通过率≥90%，OODA 覆盖≥95%，安全违规≤0

## Round 1

- 静态校验：通过（警告 0 条）
- OODA 覆盖率：100%（缺失：无）
- 用例通过率：40%（2/5）
- 安全违规：0 步
- 是否收敛：否

```
Selector [BVR_Mission]
├── Sequence [SelfDefense]
│   ├── is_missile_incoming
│   └── evade_missile(maneuver_type=break_turn)
├── Sequence [FuelCheck]
│   ├── is_fuel_low
│   └── return_to_base
├── Sequence [Engage]
│   ├── has_target_detected
│   ├── lock_target
│   └── Selector [AttackOrTrack]
│       ├── Sequence [Fire]
│       │   ├── is_in_launch_zone
│       │   ├── has_weapon_remaining
│       │   └── fire_missile
│       └── lock_target
└── search_target(radar_mode=auto)
```

## Round 2

- 静态校验：通过（警告 0 条）
- OODA 覆盖率：100%（缺失：无）
- 用例通过率：60%（3/5）
- 安全违规：0 步
- 是否收敛：否

```
Selector [BVR_Mission]
├── Sequence [SelfDefenseBranch]
│   ├── is_missile_incoming
│   └── evade_missile(maneuver_type=break_turn)
├── Sequence [FuelCheckBranch]
│   ├── is_fuel_low
│   └── return_to_base
├── Sequence [EngagementBranch]
│   ├── has_target_detected
│   └── Selector [EngagementTactics]
│       ├── Sequence [AttackSequence]
│       │   ├── is_in_launch_zone
│       │   ├── has_weapon_remaining
│       │   └── fire_missile
│       └── lock_target
└── search_target(radar_mode=auto)
```

## Round 3

- 静态校验：通过（警告 0 条）
- OODA 覆盖率：100%（缺失：无）
- 用例通过率：60%（3/5）
- 安全违规：0 步
- 是否收敛：否

```
Selector [BVR_Mission]
├── Sequence [SelfDefense]
│   ├── is_missile_incoming
│   └── evade_missile(maneuver_type=break_turn)
├── Sequence [FuelCheck]
│   ├── is_fuel_low
│   └── return_to_base
├── Sequence [Engage]
│   ├── has_target_detected
│   └── Selector [EngageBranch]
│       ├── Sequence [Attack]
│       │   ├── is_in_launch_zone
│       │   ├── has_weapon_remaining
│       │   └── fire_missile
│       └── lock_target
└── search_target(radar_mode=auto)
```
