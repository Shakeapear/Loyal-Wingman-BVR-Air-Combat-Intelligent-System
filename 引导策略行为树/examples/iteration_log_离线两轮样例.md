# 迭代收敛日志（D2.5-3）

- 时间：2026-09-17 23:29:22
- provider：mock（model=离线 mock）
- 任务：超视距空战拦截：雷达搜索并锁定目标，进入攻击区后发射中距弹（双发间隔≥4 s），导弹来袭时优先规避，燃油不足时返航，全程满足安全包线约束
- 收敛判据：通过率≥90%，OODA 覆盖≥95%，安全违规≤0

## Round 1

- 静态校验：通过（警告 0 条）
- OODA 覆盖率：75%（缺失：is_fuel_low、evade_missile）
- 用例通过率：100%（5/5）
- 安全违规：0 步
- 是否收敛：否

```
Selector [BVR_Mission]
├── Sequence [Human_Command_Response]
│   ├── has_human_command
│   └── navigate_to_waypoint(altitude_m=8000, speed_mps=250, heading_deg=0)
├── Sequence [BVR_Engagement]
│   ├── Selector [Detection]
│   │   ├── Sequence [Track_Target]
│   │   │   ├── has_target_detected
│   │   │   └── lock_target
│   │   └── search_target(radar_mode=auto)
│   └── Selector [Engage_or_Track]
│       ├── Sequence [Launch]
│       │   ├── is_in_launch_zone
│       │   └── Sequence [Fire_Control]
│       │       ├── has_weapon_remaining
│       │       └── fire_missile
│       └── lock_target
└── Sequence [Default_Patrol]
    ├── search_target(radar_mode=auto)
    └── maintain_formation
```

## Round 2

- 静态校验：通过（警告 0 条）
- OODA 覆盖率：100%（缺失：无）
- 用例通过率：100%（5/5）
- 安全违规：0 步
- 是否收敛：是

```
Selector [BVR_Mission]
├── Sequence [Human_Command_Response]
│   ├── has_human_command
│   └── navigate_to_waypoint(altitude_m=8000, speed_mps=250, heading_deg=0)
├── Sequence [Self_Defense]
│   ├── is_missile_incoming
│   └── evade_missile(maneuver_type=break_turn)
├── Sequence [Fuel_Return]
│   ├── is_fuel_low
│   └── return_to_base
├── Sequence [BVR_Engagement]
│   ├── Selector [Detection]
│   │   ├── Sequence [Track_Target]
│   │   │   ├── has_target_detected
│   │   │   └── lock_target
│   │   └── search_target(radar_mode=auto)
│   └── Selector [Engage_or_Track]
│       ├── Sequence [Launch]
│       │   ├── is_in_launch_zone
│       │   └── Sequence [Fire_Control]
│       │       ├── has_weapon_remaining
│       │       └── fire_missile
│       └── lock_target
└── Sequence [Default_Patrol]
    ├── search_target(radar_mode=auto)
    └── maintain_formation
```
