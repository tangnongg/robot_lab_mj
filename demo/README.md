# SafeFall Demo

End‑to‑end demonstration of the SafeFall protective control framework on
the **Unitree G1** humanoid robot in MuJoCo simulation.

## Pipeline

```
 ┌──────────┐    ┌──────────┐    ┌──────────┐
 │ WALKING  │───▶│ FALLING  │───▶│ STANDING │
 │ velocity │    │ safefall │    │  HoST    │
 │ policy   │    │ policy   │    │ policy   │
 └──────────┘    └──────────┘    └──────────┘
  predictor       mitigation      stand‑up
  monitors        maneuvers       recovery
```

### Phase 1 — WALKING
G1 walks forward at 0.8 m/s under a velocity‑tracking PPO policy.
A lightweight GRU fall predictor monitors proprioceptive observations
(63‑D: pelvis orientation, angular velocity, joint positions/velocities).
After ~200 steps an external push is injected to trigger a fall.

### Phase 2 — FALLING
When the predictor detects an unavoidable fall (3 consecutive frames
above threshold), control switches to the SafeFall mitigation policy.
The robot executes learned protective behaviors — retracting limbs,
orienting the torso to absorb impact on robust body regions.

### Phase 3 — STANDING
After ground impact the robot state is transferred to the HoST (Humanoid
Stand‑up Transformer) environment, where a separate PPO policy brings
the robot back to its feet.

## Dependencies

All policies must be pre‑trained and saved as PyTorch checkpoints:

| Component | Checkpoint |
|-----------|-----------|
| Velocity policy | `logs/rsl_rl/g1_velocity/…/model_1999.pt` |
| SafeFall policy | `logs/rsl_rl/safefall_g1/…/model_4998.pt` |
| HoST policy | `logs/rsl_rl/g1_host_ground/…/model_900.pt` |
| Fall predictor | `models/fall_predictor.pt` |

Update the paths in `policies.py` if your checkpoints are at different
locations.

## Usage

```bash
conda activate env_mjlab_e
cd /home/tangl/myProj/robot_lab_mj

# CPU (slower, works everywhere)
python -m demo.main --device cpu

# GPU (faster)
python -m demo.main --device cuda:0
```

## Controls

| Key | Action |
|-----|--------|
| **Esc** | Stop the demo |
| Left drag | Rotate camera |
| Right drag | Pan camera |
| Scroll | Zoom |
| Double-click | Select body to track |

## Files

```
demo/
├── main.py        # Entry point
├── runner.py      # Three-phase state machine
├── policies.py    # Policy loading utilities
└── README.md      # This file
```
