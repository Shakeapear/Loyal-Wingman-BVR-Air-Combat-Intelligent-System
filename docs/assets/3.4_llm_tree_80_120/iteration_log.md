# 迭代收敛日志（D2.5-3）

- 时间：2026-11-03 22:50:37
- provider：deepseek（model=deepseek-flash）
- 场景：bvr80-120（max_steps=600）
- 任务：超视距空战拦截：雷达搜索并锁定目标，进入攻击区后发射中距弹（双发间隔≥4 s），导弹来袭时优先规避，燃油不足时返航，全程满足安全包线约束。验证场景：80–120 km、600 步；需要远距搜索/闭合，并在 NEZ 内发射。
- 契约版本：{'frame': 'frame_v1', 'csv': 'csv_v1', 'info': 'info_v1', 'result_record': 'record_v1', 'test_case': 'test_case_v1', 'bt_schema': 'bt_v1'}
- 收敛判据：通过率≥90%，OODA 覆盖≥95%，安全违规≤0

## Round 1

- 静态校验：通过（警告 0 条）
- OODA 覆盖率：88%（缺失：fire_missile）
- 用例通过率：80%（4/5）
- 安全违规：0 步
- 是否收敛：否

```
Selector [BVR_Mission]
├── Sequence [Defend_incoming_missile]
│   ├── is_missile_incoming
│   └── evade_missile(maneuver_type=defensive_spiral)
├── Sequence [Fuel_RTB]
│   ├── is_fuel_low
│   └── return_to_base
├── Sequence [Engage_target]
│   ├── has_target_detected
│   ├── lock_target
│   └── Selector [Attack_decision]
│       ├── Sequence [Launch_when_in_NEZ]
│       │   ├── is_in_launch_zone
│       │   ├── is_in_nez
│       │   ├── has_weapon_remaining
│       │   └── execute_maneuver(maneuver=launch_and_crank, parameters={'offset_deg': 35, 'speed_mps': 250, 'duration_s': 6})
│       ├── Sequence [Crank_in_launch_zone]
│       │   ├── is_in_launch_zone
│       │   └── execute_maneuver(maneuver=crank, parameters={'offset_deg': 45, 'speed_mps': 260, 'duration_s': 5})
│       └── Sequence [Close_and_track]
│           ├── execute_maneuver(maneuver=intercept_closure, parameters={'offset_deg': 0, 'speed_mps': 300, 'duration_s': 8})
│           └── execute_maneuver(maneuver=lock_and_track, parameters={'offset_deg': 10, 'duration_s': 4})
└── Sequence [Long_range_search]
    ├── search_target(radar_mode=auto)
    └── execute_maneuver(maneuver=search_pattern, parameters={'offset_deg': 20, 'speed_mps': 280, 'duration_s': 8})
```

## Round 2

- 静态校验：通过（警告 0 条）
- OODA 覆盖率：100%（缺失：无）
- 用例通过率：100%（5/5）
- 安全违规：0 步
- 是否收敛：是

```
Selector [BVR_Mission]
├── Sequence [SelfDefense]
│   ├── is_missile_incoming
│   └── evade_missile(maneuver_type=defensive_spiral)
├── Sequence [FuelRTB]
│   ├── is_fuel_low
│   └── return_to_base
├── Sequence [Attack]
│   ├── has_target_detected
│   └── Selector [Engagement]
│       ├── Sequence [Launch]
│       │   ├── is_in_launch_zone
│       │   ├── is_in_nez
│       │   ├── has_weapon_remaining
│       │   └── fire_missile
│       ├── Sequence [CrankIfInNEZ]
│       │   ├── is_in_nez
│       │   ├── lock_target
│       │   └── execute_maneuver(maneuver=crank)
│       └── Sequence [CloseAndTrack]
│           ├── lock_target
│           └── execute_maneuver(maneuver=intercept_closure)
└── Sequence [SearchPatrol]
    ├── search_target(radar_mode=auto)
    └── execute_maneuver(maneuver=search_pattern)
```
