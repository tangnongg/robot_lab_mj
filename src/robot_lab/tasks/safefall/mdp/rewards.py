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
# Approximated via qfrc_external (body wrench in joint space).
# f_thresh is set to 0.5 * τ̄ (mechanical load capacity).
# ---------------------------------------------------------------------------


def reward_joint_reaction(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    threshold_fraction: float = 0.5,
) -> torch.Tensor:
    """Joint reaction force penalty — proxy for paper Eq.4.

    Uses qfrc_external (J^T × xfrc_applied contribution to joint space)
    as a proxy for joint reaction forces. Penalizes when external wrench
    contributions exceed a fraction of the joint's mechanical load capacity.
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


# ---------------------------------------------------------------------------
# Proxy rewards (compensate for missing r_contact)
# ---------------------------------------------------------------------------


def reward_protect_head(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward keeping base height away from ground impact.

    Core SafeFall objective: protect vulnerable components by keeping
    the base (pelvis) elevated relative to the ground during fall.
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(base_height, min=0.0, max=0.8)


def reward_body_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the robot for orienting to land on robust body parts.

    The paper shows the robot learns to rotate to land on torso/back
    rather than head/hands. Penalizes head-down orientation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    gravity_proj = asset.data.projected_gravity_b
    head_down_penalty = torch.clamp(gravity_proj[:, 2], min=0.0)
    return -head_down_penalty


def reward_energy_absorption(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward controlled deceleration (energy absorption through joints).

    The paper emphasizes distributing impact over time. We reward
    negative work done by joints (energy absorption).
    """
    asset: Entity = env.scene[asset_cfg.name]
    power = asset.data.qfrc_actuator * asset.data.joint_vel
    absorption = torch.clamp(-power, min=0.0)
    return torch.sum(absorption, dim=-1) * 0.001


def reward_base_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize high base linear velocity at impact (reduce impact force)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=-1)
