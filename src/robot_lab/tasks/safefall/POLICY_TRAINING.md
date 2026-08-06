# SafeFall Policy Training

This document covers the protective policy only. The fall predictor checkpoint can
be reused; its architecture and training do not need to change.

## What the policy environment implements

- 200 Hz PD control and physics, 50 Hz policy inference.
- A finite 40-step (0.8 s) episode with no early successful-fall termination.
- Five-frame deployable actor observations and an asymmetric privileged critic.
- Paper Eq. 3 contact cost with component weights `{1000, 1, 0.5}`.
- Paper Eq. 4 joint reaction force from MuJoCo `cfrc_int`.
- Paper Eq. 5 external joint torque from `qfrc_constraint`, excluding motor output.
- Peak load sampling at every 200 Hz physics substep.
- Post-impact settling: after terrain contact, track the pre-impact root/joint
  motion peaks and reward a sustained low-velocity state.
- Stage I without self-collision and Stage II with the full collision model.
- Per-joint absolute target clipping and Table II randomization, except restitution.
- Numerical-state reset, bounded non-physical cost tails, and a capped exploration
  standard deviation for long-run simulator stability.

## Stage I

Stage I samples broad, omnidirectional falling states. The reset distribution starts
without ground contact and guarantees that impact occurs inside the short episode.

```bash
PYTHONPATH=src python -m mjlab.scripts.train Mjlab-SafeFall-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.run-name stage1 \
  --agent.logger tensorboard
```

Before a long run, calibrate the three impact weights on the target machine:

```bash
PYTHONPATH=src python -m robot_lab.tasks.safefall.calibrate_reward_weights \
  --num-envs 256 --num-episodes 5 --device cuda:0
```

The script targets an average contribution of about one reward unit for each active
impact term per policy step. Keep the paper's relative component sensitivity inside
`r_contact`; tune only the three global weights.

## Stage II State Bank

Stage II requires exact simulator state at predictor-positive frames. Predictor-only
63-D observations do not contain root height, linear velocity, or yaw, so old
trajectory files without `state_schema_version: 2` cannot produce a valid bank.
Recollecting trajectories does not require retraining the predictor.

```bash
PYTHONPATH=src python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
  --output data/fall_trajs_v2 \
  --num-trajs 81920 \
  --num-envs 256 \
  --device cuda:0 \
  --policy-checkpoint logs/rsl_rl/g1_velocity/<run>/model_<iteration>.pt \
  --policy-task Mjlab-Velocity-Flat-Unitree-G1 \
  --rough

PYTHONPATH=src python -m robot_lab.tasks.safefall.fall_predictor.prepare_stage2_states \
  --traj-dir data/fall_trajs_v2 \
  --predictor models/fall_predictor.pt \
  --output models/stage2_state_bank.pt \
  --device cuda:0
```

## Stage II

Resume Stage II from the final Stage I checkpoint. Both tasks intentionally share
the `safefall_g1` experiment directory.

```bash
PYTHONPATH=src python -m mjlab.scripts.train Mjlab-SafeFall-StageII-G1 \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run '.*_stage1' \
  --agent.load-checkpoint 'model_4999.pt' \
  --agent.max-iterations 5000 \
  --agent.run-name stage2 \
  --agent.logger tensorboard
```

## Reproduction Limits

The paper does not publish `w_c`, `w_j`, `w_e`, per-joint mechanical force
thresholds, or MLP layer widths. This implementation uses measured reward scaling,
a 500 N reaction-force threshold, and the migrated network widths. MuJoCo-Warp also
differs from the paper's original simulator. Restitution randomization remains
unimplemented because mjlab does not expose a per-environment pair restitution
randomizer. These prevent bit-for-bit reproduction, but no longer remove or invert
the paper's damage signals.

## Acceptance Video

Record a finite deterministic rollout from a local checkpoint without launching an
interactive viewer:

```bash
PYTHONPATH=src python -m robot_lab.tasks.safefall.record_play_video \
  --checkpoint logs/rsl_rl/safefall_g1/<run>/model_<iteration>.pt \
  --frames 200 --device cuda:0
```
