"""Custom reward functions for SafeFall damage mitigation task.

Implements the damage-aware reward from the SafeFall paper (Meng et al., 2025):

  r_total = r_impact + r_regulation

  r_impact = w_c * r_contact + w_j * r_joint + w_e * r_torque

Where:
- r_contact (Eq.3): Per-link contact forces with component heterogeneity weights.
  Not implemented (requires contact sensor with per-body force data).
- r_joint (Eq.4): Joint reaction force penalty with per-joint mechanical thresholds.
  Approximated via qfrc_external (body wrench contribution to joint space).
- r_torque (Eq.5): Normalized torque ratio [|τ_i|/τ̄_i − 1]_+², preventing
  actuator saturation and mechanical stress.

Proxy rewards (not in paper, compensate for missing r_contact):
  protect_head, body_orientation, energy_absorption, base_lin_vel

Migrated from IsaacLab to mjlab. Key API changes:
- asset.data.applied_torque → asset.data.qfrc_actuator
- asset.data.root_pos_w → asset.data.root_link_pos_w
- asset.data.root_lin_vel_w → asset.data.root_link_lin_vel_w
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ---------------------------------------------------------------------------
# Per-joint max rated torque (τ̄_i) for the G1 — derived from actuator
# effort_limit values in g1_constants.py.  Joint order matches the mjlab
# G1 MJCF (29 DoF enumerated by Entity.joint_names).
# ---------------------------------------------------------------------------
# left_hip_pitch/roll/yaw, left_knee, left_ankle_pitch/roll,
# right_hip_pitch/roll/yaw, right_knee, right_ankle_pitch/roll,
# waist_yaw/roll/pitch,
# left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll/pitch/yaw,
# right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/pitch/yaw
_MAX_TORQUE = torch.tensor(
    [
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,   # left leg
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,   # right leg
        88.0, 50.0, 50.0,                        # waist
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, # left arm
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, # right arm
    ]
)

# ---------------------------------------------------------------------------
# Component heterogeneity weights (paper §III-D, Eq.3)
# geom name → vulnerability weight w_s ∈ {1000, 1, 0.5}
# ---------------------------------------------------------------------------
# High:   head, hands                                 → w_s = 1000
# Medium: shanks, shoulders, feet                      → w_s = 1
# Low:    torso, thighs, elbows, pelvis, hips, etc.   → w_s = 0.5
# ---------------------------------------------------------------------------
_VULNERABILITY_HIGH = 1000.0
_VULNERABILITY_MED = 1.0
_VULNERABILITY_LOW = 0.5

_GEOM_VULNERABILITY: dict[str, float] = {
    "head_collision": _VULNERABILITY_HIGH,
    "left_hand_collision": _VULNERABILITY_HIGH,
    "right_hand_collision": _VULNERABILITY_HIGH,
    # medium
    "left_shin_collision": _VULNERABILITY_MED,
    "right_shin_collision": _VULNERABILITY_MED,
    "left_shoulder_yaw_collision": _VULNERABILITY_MED,
    "right_shoulder_yaw_collision": _VULNERABILITY_MED,
    # low
    "torso_collision": _VULNERABILITY_LOW,
    "pelvis_collision": _VULNERABILITY_LOW,
    "left_thigh_collision": _VULNERABILITY_LOW,
    "right_thigh_collision": _VULNERABILITY_LOW,
    "left_hip_collision": _VULNERABILITY_LOW,
    "right_hip_collision": _VULNERABILITY_LOW,
    "left_elbow_yaw_collision": _VULNERABILITY_LOW,
    "right_elbow_yaw_collision": _VULNERABILITY_LOW,
    "left_wrist_collision": _VULNERABILITY_LOW,
    "right_wrist_collision": _VULNERABILITY_LOW,
    "left_linkage_brace_collision": _VULNERABILITY_LOW,
    "right_linkage_brace_collision": _VULNERABILITY_LOW,
}

# ---------------------------------------------------------------------------
# Paper r_contact (Eq. 3): component‑heterogeneity contact force penalty
#   r_contact = (1/N) Σ I{c_i}·w_{s,i}·[f_{contact,i} − m_i·g]_+²
#             + α · max_i { I{c_i}·w_{s,i}·[f_{contact,i} − m_i·g]_+² }
# where N = Σ I{c_i} (active contacts) and α = 0.3
# ---------------------------------------------------------------------------


def reward_contact_force(
    env: ManagerBasedRlEnv,
    alpha: float = 0.3,
    sensor_name: str = "body_ground_contact",
) -> torch.Tensor:
    """Per‑link contact force penalty with component heterogeneity weights.

    Reads per‑geom contact forces from a ContactSensor and applies
    vulnerability weights w_s ∈ {1000, 1, 0.5} per the paper (head/hands
    → 1000, shanks/shoulders → 1, torso/thighs/elbows → 0.5).

    Adjacent‑link collisions are excluded by the sensor pattern (only
    links versus terrain, not link‑versus‑link).
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force is None or sensor.data.found is None:
        return torch.zeros(env.num_envs, device=env.device)

    # ContactData tensors: [B, P], [B, P, 3] (no history/slots when
    # history_length=1, num_slots=1, reduce="none").
    force = sensor.data.force  # [B, P, 3]
    found = sensor.data.found  # [B, P]
    contact_mask = (found > 0.0).float()  # [B, P]

    # Build per‑geom vulnerability weight tensor.
    w_s = torch.tensor(
        [_GEOM_VULNERABILITY.get(n, _VULNERABILITY_LOW)
         for n in sensor.primary_names],
        device=env.device,
    )  # (N_geoms,)

    # Body mass vector (kg) — indexed via geom→body lookup on CPU.
    asset: Entity = env.scene["robot"]
    mjm = env.sim.mj_model
    g_val = mjm.opt.gravity[2]  # −9.81, take abs for loading
    body_mass = torch.tensor(
        [mjm.body_mass[mjm.geom_bodyid[g]]
         for g, _ in enumerate(sensor.primary_names)],
        device=env.device,
    )  # (N_geoms,)

    # Contact force magnitude.
    f_contact = torch.norm(force, dim=-1)  # (B, N_geoms)
    m_g = body_mass * abs(g_val)  # (N_geoms,) gravitational loading

    # Clip to penalty: only force excess over gravitational loading.
    excess = torch.clamp(f_contact - m_g.unsqueeze(0), min=0.0)  # (B, N_geoms)

    weighted = contact_mask * w_s.unsqueeze(0) * excess ** 2  # (B, N_geoms)

    N_active = contact_mask.sum(dim=-1).clamp(min=1)  # (B,)

    r_avg = weighted.sum(dim=-1) / N_active  # Eq.3 first term
    r_peak = weighted.max(dim=-1)[0]  # Eq.3 second term (max over i)

    return -(r_avg + alpha * r_peak)


# ---------------------------------------------------------------------------
# Paper r_torque (Eq. 5):  r_torque = Σ_j [|τ_j| / τ̄_j − 1]_+²
# ---------------------------------------------------------------------------


def reward_joint_torques(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Normalized torque ratio penalty matching paper Eq.5.

    Penalizes joint torques that exceed the actuator's maximum rated torque.
    τ̄_i is the actuator effort_limit per joint.
    """
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator  # (N, 29) actuator force in joint space
    tau_bar = _MAX_TORQUE.to(tau.device)  # (29,)
    ratio = torch.abs(tau) / tau_bar.clamp(min=1.0)  # (N, 29)
    penalty = torch.clamp(ratio - 1.0, min=0.0)
    return torch.sum(penalty ** 2, dim=-1)


# ---------------------------------------------------------------------------
# Paper r_joint (Eq. 4):  r_joint = Σ_j ‖f_{joint,j} − f_thresh,j‖²
#
# f_{joint,j} is the joint-reaction force that maintains kinematic
# constraints between adjacent links during impact propagation.
# MuJoCo stores this in efc_force, but the Warp backend does not
# expose efc_force to Python (shape is always (0,)).  We use
# qfrc_external (J^T × body wrench in joint space) as a proxy —
# it captures the joint-space contribution of external body-level
# wrenches (xfrc_applied), which are the dominant loading component
# during falls.
#
# f_thresh,j is set to 0.5 × τ̄_j, where τ̄_j is the actuator's
# maximum rated torque (effort_limit).  This approximates the
# joint's mechanical load capacity: for typical robot actuators,
# the sustainable static structural load is ~50% of the peak
# electromagnetic torque before bearing / gear degradation.
# ---------------------------------------------------------------------------

def reward_joint_reaction(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    threshold_fraction: float = 0.5,
) -> torch.Tensor:
    """Joint reaction force penalty — proxy for paper Eq.4.

    Proxy: replaces f_{joint} with qfrc_external (body wrench in
    joint space).  Penalises when this exceeds half the actuator's
    maximum rated torque (mechanical load capacity estimate).
    """
    asset: Entity = env.scene[asset_cfg.name]
    f_ext = torch.abs(asset.data.qfrc_external)  # (N, 29)
    tau_bar = _MAX_TORQUE.to(f_ext.device)
    f_thresh = tau_bar * threshold_fraction  # mechanical load capacity
    penalty = torch.clamp(f_ext - f_thresh, min=0.0)
    return torch.sum(penalty ** 2, dim=-1)


# ---------------------------------------------------------------------------
# Regularization rewards (r_regulation)
# ---------------------------------------------------------------------------


def reward_action_rate(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Penalize rapid action changes for smooth protective motions."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1
    )


def reward_joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive joint velocities."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=-1)


def reward_joint_acc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=-1)


def reward_joint_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions near limits."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    lower = asset.data.soft_joint_pos_limits[..., 0]
    upper = asset.data.soft_joint_pos_limits[..., 1]
    below = torch.clamp(lower - pos, min=0.0)
    above = torch.clamp(pos - upper, min=0.0)
    return torch.sum(below + above, dim=-1)


