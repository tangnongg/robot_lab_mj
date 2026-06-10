# SafeFall Paper Review — Implementation Gap Analysis

## Paper: *SafeFall: Learning Protective Control for Humanoid Robots* (Meng et al., 2025)

> **Status**: Review completed 2026-06-10. All actionable gaps have been fixed.
> The remaining unfixable gaps are documented in Section 4.

---

## 1. Critical Gaps Found

### 1.1 Episode Length — **MISMATCH**

| | Value |
|---|---|
| Paper | "fixed episode length of **40**" (40 steps × 0.02s = **0.8s**) |
| Implementation | `episode_length_s = 2.0` (100 steps) |

**Paper rationale**: "To address the temporal credit assignment problem inherent in sparse impact signals, we restrict training to a fixed, short episode length — the detection of an unavoidable fall to the completion of ground impact."

**Fix priority**: 🔴 P0 — This is a core design decision in the paper. Using 100 steps instead of 40 defeats the purpose of short episodes for temporal credit assignment.

### 1.2 Reward Function — **MAJOR SIMPLIFICATION**

The paper defines `r_total = r_impact + r_regulation` where:

**r_impact = w_c · r_contact + w_j · r_joint + w_e · r_torque**

| Term | Paper Definition | Implementation | Gap |
|------|-----------------|----------------|-----|
| r_contact (Eq.3) | Per-link contact forces with component heterogeneity weights (w_s ∈ {1000, 1, 0.5}), adjacent-link collision filtering, average+peak force balance | **Not implemented** | 🔴 |
| r_joint (Eq.4) | Joint reaction force penalty: `sum(‖f_{joint,i} − f_{thresh,i}‖²)` with per-joint mechanical load thresholds | **Not implemented** (mjlab lacks direct joint reaction force data) | 🔴 |
| r_torque (Eq.5) | Normalized torque ratio: `sum([τ_i / τ̄_i − 1]_+²)` where τ̄_i is max rated torque per joint | `sum(τ²)` — simple squared torque, **no normalization** | 🟡 |

**Proxy rewards added** that are NOT in the paper:
- `protect_head` (+5.0): Rewards base height
- `body_orientation` (+3.0): Penalizes head-down
- `energy_absorption` (+2.0): Rewards joint energy absorption
- `base_lin_vel` (-0.01): Penalizes base linear velocity

These proxy rewards partially compensate for the missing paper rewards but don't capture component heterogeneity or joint mechanical constraints.

### 1.3 Asymmetric Actor-Critic — **NOT IMPLEMENTED**

| | Actor | Critic |
|---|---|---|
| Paper | Deployable sensors only (IMU + joint encoders) | Actor obs + privileged: global root pos/vel, CoM |
| Implementation | deployable sensors | **Same as actor (symmetric)** |

The paper uses asymmetric observations to "exploit rich supervision" during training. Our symmetric implementation limits value function accuracy.

### 1.4 Domain Randomization — **SEVERELY REDUCED**

Paper (Table II) prescribes:

| Parameter | Paper Range | Implementation |
|-----------|-------------|----------------|
| Friction | U(0.3, 1.0) | ❌ Missing |
| Restitution | U(0.0, 0.5) | ❌ Missing |
| Base mass offset | U(−1.0, 3.0) kg | ❌ Missing |
| Base CoM offset | x,y ~ U(−0.05, 0.05), z ~ U(−0.01, 0.01) | ❌ Missing |
| Joint stiffness scale | logU(0.7, 1.5) | ❌ Missing |
| Joint damping scale | logU(0.5, 3.0) | ❌ Missing |
| Joint position limits | N(0, 0.02) | ❌ Missing |
| Observation noise (all) | U(−0.05, 0.05), U(−0.01, 0.01), U(−1.5, 1.5), U(−0.2, 0.2), U(−0.05, 0.05) | ✅ Present |

**Impact**: Without DR, the policy will not transfer to real hardware. This is critical for sim-to-real deployment.

### 1.5 Two-Stage Curriculum — **STAGE II MISSING**

The paper uses a critical two-stage curriculum:
- **Stage I**: Random falling configurations (random poses, heights, velocities) → rapid exploration
- **Stage II**: Realistic falling states sampled from the fall predictor's output → refine for real-world distributions

Only Stage I is implemented. Stage II requires the fall predictor (separate component) to be trained first.

### 1.6 Observation — **Extra Term Added**

| Term | Paper | Implementation |
|------|-------|----------------|
| base_height | ❌ Not in paper | ✅ Included |

**Rationale**: The paper explicitly says "global root position" is privileged info only available to the critic. Adding base_height to the actor is a pragmatic addition (it's derivable in simulation) but differs from the paper's deployable-only observation design.

---

## 2. What Matches Correctly

| Element | Status |
|---------|--------|
| PD controller at 200Hz, policy at 50Hz (timestep=0.005, decimation=4) | ✅ |
| Observation stacking (history_length=5) | ✅ |
| PPO hyperparameters (lr=1e-3, adaptive, 5 epochs, 4 mini-batches) | ✅ |
| GRU hidden size 64 and observation terms from paper | N/A (fall predictor separate) |
| Reset to random falling states (Stage I) | ✅ |
| Joint velocity termination (200 rad/s) | ✅ |
| 29-DOF G1 robot | ✅ |

---

## 3. Fix Implementation Plan

### Fix 1: Episode Length → 0.8s
```python
# env_cfgs.py
cfg.episode_length_s = 0.8  # 40 steps at 50Hz
```

### Fix 2: Normalize r_torque with per-actuator max torque
Replace `sum(τ²)` with `sum([|τ| / τ̄ − 1]_+²)`.
- Derive τ̄ from actuator `effort_limit` values per joint group
- G1 actuator max torques: hip=88Nm, knee=139Nm, ankle=50Nm (2×25), shoulder/elbow=25Nm, waist=25×2, wrist=5Nm

### Fix 3: Asymmetric Critic
Add a `"critic"` observation group that includes:
- All actor observations (without noise)
- Root base position (privileged)
- Root linear velocity (privileged)
- Center of mass position (privileged, if available)

### Fix 4: Domain Randomization
Add startup/reset events for:
- Geom friction randomization (via mjlab DR `randomize_geom_friction`)
- Body mass randomization (via `pseudo_inertia` as go2w does)
- Joint stiffness/damping randomization (via mjlab DR for actuators)

### Fix 5: r_joint proxy via qfrc_external
Use `qfrc_external` (body wrench contribution to joint space) as a proxy for joint reaction forces, with per-group effort limits as thresholds.

---

## 4. Limitations That Cannot Be Fixed

1. **r_contact with component heterogeneity**: Requires per-link contact force sensors with body-name mapping — not directly available in mjlab without ContactSensor + ContactMatch configuration. The paper uses this to distinguish high-vulnerability (head/camera, hands) from low-vulnerability (torso, thighs) contacts.
2. **Stage II curriculum**: Requires the fall predictor (GRU network trained on 81,920 fall trajectories), which is a separate component.
3. **True joint reaction forces**: MuJoCo exposes `qfrc_external` (J^T × body wrench) but not constraint reaction forces at individual joints. Our `reward_joint_reaction` uses `qfrc_external` as a proxy.
4. **Base mass offset U(-1.0, 3.0)**: The `pseudo_inertia` DR function uses log-uniform scaling which preserves positive-definiteness but doesn't directly support additive mass offsets.

---

## 5. Fixes Applied (2026-06-10)

| Fix | Before | After |
|-----|--------|-------|
| Episode length | 2.0s (100 steps) | **0.8s (40 steps)** — matches paper |
| r_torque formula | `sum(τ²)` | `sum([|τ|/τ̄ − 1]_+²)` — paper Eq.5 with per-joint max effort limits |
| r_joint | Not implemented | `reward_joint_reaction` using `qfrc_external` as proxy for paper Eq.4 |
| Asymmetric critic | Symmetric (same obs) | Critic gets privileged root pos, root lin vel, CoM — noise-free |
| Domain randomization | Observation noise only | Added: geom friction U(0.3,1.0), pseudo_inertia scaling, PD gain logU randomization |
| Reward count | 9 terms | **10 terms** (added `joint_reaction` w=-5e-7) |
