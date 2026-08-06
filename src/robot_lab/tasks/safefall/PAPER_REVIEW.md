# SafeFall Paper Reproduction Audit

This is the current implementation audit for *SafeFall: Learning Protective
Control for Humanoid Robots* (Meng et al., 2025). For executable commands, see
[`POLICY_TRAINING.md`](POLICY_TRAINING.md).

## Implemented

| Paper element | Current implementation |
|---|---|
| Control rates | 200 Hz physics/PD control and 50 Hz policy inference |
| Episode | Finite 40-step (0.8 s) horizon |
| Observations | Five-frame actor history and asymmetric privileged critic |
| Contact damage, Eq. 3 | Per-component force, vulnerability weights, average and peak terms |
| Joint reaction damage, Eq. 4 | Peak parent-child interaction force from `cfrc_int` at every physics substep |
| External torque damage, Eq. 5 | Peak normalized `qfrc_constraint` torque at every physics substep |
| Stage I | Broad omnidirectional falling resets without self-collision |
| Stage II | Exact predictor-positive simulator states with full collision |
| Domain randomization | Friction, pelvis mass/CoM, PD gains, joint limits, and observation noise |

All three impact functions return non-negative costs and use negative reward
weights. Contact geom IDs are resolved through the runtime entity indexing; list
positions are never treated as MuJoCo geom IDs. Physics-substep accumulation is
required because a 200 Hz impact peak can occur between two 50 Hz policy steps.

## Stage II Data Requirement

The old 63-D predictor observations are insufficient to restore a simulator state:
they omit root height, root linear velocity, and yaw. Stage II therefore requires
new trajectory files with `state_schema_version=2`, including root state, absolute
joint position, and joint velocity. The existing predictor checkpoint can be
reused; only its input trajectories need to be recollected.

## Remaining Reproduction Limits

The paper does not publish the three global impact weights, per-joint mechanical
force thresholds, or policy MLP widths. The implementation calibrates reward
magnitudes empirically, uses a 500 N reaction-force threshold, and retains the
migrated network widths. MuJoCo-Warp is also not the paper's original simulator,
and mjlab currently has no per-environment contact-pair restitution randomizer.
These differences prevent bit-for-bit reproduction and require evaluation after
training, but the paper's principal policy inputs, curriculum, and damage signals
are represented.
