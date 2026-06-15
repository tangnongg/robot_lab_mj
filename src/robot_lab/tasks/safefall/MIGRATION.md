# SafeFall Migration: IsaacLab → mjlab

## 1. Overview

This document describes the migration of the SafeFall protective control training environment from IsaacLab (GPU PhysX-based) to mjlab (MuJoCo-Warp-based).

**Source**: `/home/tangl/myProj/safefall_isaaclab` (conda env: `env_isaaclab`)
**Target**: `/home/tangl/myProj/robot_lab_mj/src/robot_lab/tasks/safefall` (conda env: `env_mjlab_e`)

> **Note**: The **Fall Predictor** (GRU-based binary classifier, paper §III-C) has been
> implemented from scratch in `fall_predictor/`. See [FALL_PREDICTOR.md](FALL_PREDICTOR.md)
> for full documentation. The original IsaacLab project did not include this component.

**Task ID**: `Mjlab-SafeFall-G1`

---

## 2. Architecture Mapping

### 2.1 Framework Differences

| Component | IsaacLab | mjlab |
|-----------|----------|-------|
| Physics engine | PhysX (GPU) | MuJoCo-Warp (GPU) |
| Robot config | URDF + `ArticulationCfg` | MJCF + `EntityCfg` |
| Task registration | `gym.register()` | `register_mjlab_task()` |
| Training launcher | Hydra + `AppLauncher` | tyro CLI |
| RL config | `RslRlPpoActorCriticCfg` | `RslRlModelCfg` (separate actor/critic) |

### 2.2 Entity Data API Changes

| IsaacLab | mjlab |
|----------|-------|
| `asset.data.root_pos_w` | `asset.data.root_link_pos_w` |
| `asset.data.root_quat_w` | `asset.data.root_link_quat_w` |
| `asset.data.root_lin_vel_w` | `asset.data.root_link_lin_vel_w` |
| `asset.data.root_ang_vel_w` | `asset.data.root_link_ang_vel_w` |
| `asset.data.applied_torque` | `asset.data.qfrc_actuator` |
| `asset.write_root_state_to_sim(state, env_ids)` | Same method available (N, 13) format |
| `asset.write_joint_state_to_sim(pos, vel, env_ids)` | Same method available |

### 2.3 Config Class Changes

| IsaacLab | mjlab |
|----------|-------|
| `isaaclab.envs.ManagerBasedRLEnvCfg` | `mjlab.envs.ManagerBasedRlEnvCfg` |
| `isaaclab.scene.InteractiveSceneCfg` | `mjlab.scene.SceneCfg` |
| `isaaclab.terrains.TerrainImporterCfg` | `mjlab.terrains.TerrainEntityCfg` |
| `isaaclab.sim.RigidBodyMaterialCfg` | N/A (use MuJoCo geom friction) |
| `isaaclab.managers.EventTermCfg` | `mjlab.managers.event_manager.EventTermCfg` |
| `isaaclab.managers.RewardTermCfg` | `mjlab.managers.reward_manager.RewardTermCfg` |
| `isaaclab.managers.ObservationGroupCfg` | `mjlab.managers.observation_manager.ObservationGroupCfg` |
| `isaaclab.managers.TerminationTermCfg` | `mjlab.managers.termination_manager.TerminationTermCfg` |
| `isaaclab.managers.ObservationTermCfg` | `mjlab.managers.observation_manager.ObservationTermCfg` |
| `isaaclab.utils.noise.AdditiveUniformNoiseCfg` | `mjlab.utils.noise.UniformNoiseCfg` |
| `isaaclab.actuators.ImplicitActuatorCfg` | `mjlab.actuator.BuiltinPositionActuatorCfg` |
| `isaaclab.assets.AssetBaseCfg` (DomeLight) | N/A (no sky light in MuJoCo) |

---

## 3. File Migration Map

### Source → Target

```
safefall_isaaclab/                          robot_lab_mj/src/robot_lab/tasks/safefall/
├── safefall_env_cfg.py          →          env_cfgs.py
├── agents/rsl_rl_ppo_cfg.py    →          rl_cfg.py
├── __init__.py (gym.register)  →          __init__.py (register_mjlab_task)
└── mdp/
    ├── rewards.py              →          mdp/rewards.py
    ├── observations.py         →          mdp/observations.py
    ├── terminations.py         →          mdp/terminations.py
    ├── events.py               →          mdp/events.py
    └── __init__.py             →          mdp/__init__.py
```

### NOT Migrated (using mjlab equivalents)

- `robots/g1/g1_cfg.py` + URDF → Uses `robot_lab.asset_zoo.robots.unitree_g1.g1_constants`
- `scripts/rsl_rl/train.py` → Uses `mjlab/scripts/train.py`
- `scripts/rsl_rl/play.py` → Uses `mjlab/scripts/play.py`
- `scripts/rsl_rl/cli_args.py` → Uses tyro CLI (built into mjlab)

---

## 4. Key Migration Decisions

### 4.1 Robot Configuration

The G1 robot in mjlab uses the full 29-DOF MJCF specification (vs 23-DOF URDF in IsaacLab). This means:
- Observation dimension increased: 78 → 96 per timestep
- Joint space difference: 23 → 29 joints
- PD gains were overridden to match IsaacLab values (e.g., hip damping 5.0 vs mjlab default)

### 4.2 PD Gain Override

The mjlab G1 actuators use physically-derived PD gains from motor parameters. These were overridden to match the IsaacLab SafeFall config:

| Joint Group | Stiffness (Nm/rad) | Damping (Nm·s/rad) |
|-------------|-------------------|---------------------|
| hip         | 150               | 5                   |
| knee        | 200               | 6                   |
| ankle       | 40                | 2                   |
| shoulder    | 100               | 4                   |
| elbow       | 100               | 4                   |
| waist       | 100               | 4                   |
| wrist       | 100               | 4                   |

### 4.3 Observation Space

Seven observation terms stacked over 5 timesteps (history_length=5):

| Term | Dim | Source |
|------|-----|--------|
| pelvis_orientation | 2 | Custom (roll, pitch from quaternion) |
| base_ang_vel | 3 | Built-in `base_ang_vel` |
| projected_gravity | 3 | Built-in `projected_gravity` |
| joint_pos | 29 | Built-in `joint_pos_rel` |
| joint_vel | 29 | Built-in `joint_vel_rel` |
| last_action | 29 | Built-in `last_action` |
| base_height | 1 | Custom |

Total: 96 per step × 5 history = 480-dim observation.

### 4.4 Reward Structure

9 reward terms, matching IsaacLab weights:

| Reward | Weight | Type |
|--------|--------|------|
| torque_penalty | -2.5e-6 | Damage mitigation |
| protect_head | +5.0 | Protective behavior |
| body_orientation | +3.0 | Protective behavior |
| energy_absorption | +2.0 | Protective behavior |
| action_rate | -0.01 | Regularization |
| joint_vel | -1e-4 | Regularization |
| joint_acc | -2.5e-7 | Regularization |
| joint_pos_limits | -10.0 | Regularization |
| base_lin_vel | -0.01 | Regularization |

### 4.5 Termination Conditions

| Condition | Threshold |
|-----------|-----------|
| time_out | 2.0s episode |
| fall_completed | height < 0.15m AND velocity < 0.3 m/s |
| joint_vel_exceeded | max joint vel > 200 rad/s |

### 4.6 Reset Events

One reset event randomizes the robot into falling configurations:
- Height: [0.4, 0.9] m
- Roll/Pitch: [-0.8, 0.8] rad
- Yaw: [-π, π]
- Linear velocity: random downward ([-2, 2] horizontal, [-2, -0.5] vertical)
- Angular velocity: [-1, 1] rad/s
- Joint noise: ±0.2 rad position, ±0.5 rad/s velocity

### 4.7 Physics Material Domain Randomization

The IsaacLab `randomize_rigid_body_material` (startup event with 64 buckets for friction/restitution) was intentionally simplified. MuJoCo uses geom-level friction parameters set in the MJCF, and comprehensive material DR can be added later via `mjlab.envs.mdp.dr.material`.

---

## 5. Known Differences from IsaacLab Implementation

1. **Joint count**: 29 (mjlab MJCF) vs 23 (IsaacLab URDF) — full G1 specification
2. **Observation dimension**: 96/step vs 78/step, due to 6 extra wrist/hand joints
3. **No contact sensor**: IsaacLab version also skipped contact sensors; mjlab version continues this simplification
4. **No Stage II curriculum**: Only Stage I (random falling states) is implemented, matching IsaacLab version
5. **Symmetric actor-critic**: No privileged critic observations, matching IsaacLab version
6. **Physics backend**: MuJoCo-Warp (elliptic cone, implicitfast integrator) vs PhysX (TGS solver)

---

## 6. Usage

### Training

```bash
conda activate env_mjlab_e

# Quick test
python -m mjlab.scripts.train Mjlab-SafeFall-G1 --env.scene.num-envs 64 --agent.max-iterations 100

# Full training (4096 envs, 5000 iterations)
python -m mjlab.scripts.train Mjlab-SafeFall-G1

# Resume from checkpoint
python -m mjlab.scripts.train Mjlab-SafeFall-G1 --agent.resume True --agent.load-run <run_dir>
```

### Playback

```bash
conda activate env_mjlab_e
python -m mjlab.scripts.play Mjlab-SafeFall-G1 --agent.load-run <run_dir> --agent.load-checkpoint model_5000.pt
```

---

## 7. Verification Results

| Check | Status | Notes |
|-------|--------|-------|
| Import test | ✅ | All modules import without errors |
| Config creation | ✅ | Train and play configs create correctly |
| Task registration | ✅ | `Mjlab-SafeFall-G1` listed in registry |
| Environment instantiation | ✅ | 50-env play mode creates successfully |
| Single step | ✅ | Observations (480,), actions (29,) confirmed |
| Training loop | ✅ | 5 iterations completed, rewards computed |
| Reward functions | ✅ | All 9 reward terms produce valid values |
| Termination functions | ✅ | All 3 termination conditions active |
| Reset event | ✅ | Random falling states applied on reset |

---

## 8. Files Created

```
src/robot_lab/tasks/safefall/
├── __init__.py           # Task registration
├── env_cfgs.py           # Environment configuration factory
├── rl_cfg.py             # PPO hyperparameter configuration
├── mdp/
│   ├── __init__.py       # MDP term re-exports
│   ├── rewards.py        # 9 custom reward functions
│   ├── observations.py   # 2 custom observation functions
│   ├── terminations.py   # 2 custom termination functions
│   └── events.py         # 1 custom reset event function
└── MIGRATION.md          # This document
```

Plus modification to `src/robot_lab/tasks/__init__.py`:
```python
from . import safefall  # Added
```
