"""Custom reward functions for HoST stand-up task.

Ported from Isaac Lab to mjlab API. Body name differences:
- ``keyframe_head_link`` → ``head_link``
- ``body_pos_w`` → ``body_link_pos_w``

All functions receive ``env: ManagerBasedRlEnv`` and return a per-environment
reward tensor of shape ``(num_envs,)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _sigmoid(x: torch.Tensor, value_at_1: float) -> torch.Tensor:
    """Gaussian-like sigmoid: exp(-0.5 * (x * scale)^2)."""
    scale = np.sqrt(-2 * np.log(max(value_at_1, 1e-9)))
    return torch.exp(-0.5 * (x * scale) ** 2)


def _tolerance(
    x: torch.Tensor,
    bounds: tuple[float, float],
    margin: float = 0.0,
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """Returns 1.0 inside *bounds*, decays smoothly outside via sigmoid.

    When *margin* == 0 the function is a hard step (1.0 inside, 0.0 outside).
    """
    lower, upper = bounds
    in_bounds = (lower <= x) & (x <= upper)
    if margin == 0:
        return torch.where(in_bounds, 1.0, 0.0)
    d = torch.where(x < lower, lower - x, x - upper) / margin
    return torch.where(in_bounds, 1.0, _sigmoid(d.double(), value_at_margin).float())


# ---------------------------------------------------------------------------
# Task Rewards
# ---------------------------------------------------------------------------


def reward_orientation(
    env: "ManagerBasedRlEnv",
    phase1_height: float = 0.45,
    orientation_threshold: float = 0.99,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward upright orientation when base height is above *phase1_height*."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity_proj = asset.data.projected_gravity_b
    base_height = asset.data.root_link_pos_w[:, 2]
    reward = _tolerance(
        -gravity_proj[:, 2], (orientation_threshold, np.inf), 1.0, 0.05
    )
    return reward * (base_height > phase1_height).float()


def reward_head_height(
    env: "ManagerBasedRlEnv",
    target_head_height: float = 1.0,
    target_head_margin: float = 1.0,
    head_body_name: str = "torso_link",
    foot_body_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward head height above feet."""
    asset: Entity = env.scene[asset_cfg.name]
    body_names = list(asset.body_names)

    head_idx = body_names.index(head_body_name)
    head_height = asset.data.body_link_pos_w[:, head_idx, 2:3]

    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_indices = [body_names.index(n) for n in foot_body_names]
    feet_height = asset.data.body_link_pos_w[:, foot_indices, 2].mean(
        dim=-1, keepdim=True
    )

    relative_height = head_height - feet_height
    reward = _tolerance(
        relative_height, (target_head_height, np.inf), target_head_margin, 0.1
    )
    return reward.squeeze(-1)


# ---------------------------------------------------------------------------
# Regularization Rewards
# ---------------------------------------------------------------------------


def reward_dof_acc(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def reward_action_rate(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize changes in actions."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def reward_smoothness(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Second-order action smoothness penalty."""
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    if hasattr(env, "host_last_last_action"):
        return torch.sum(
            torch.square(action - 2 * prev + env.host_last_last_action), dim=1
        )
    return torch.sum(torch.square(action - prev), dim=1)


def reward_torques(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint torques (actuator forces in joint space)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.qfrc_actuator), dim=1)


def reward_joint_power(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint power (|vel| * |actuator torque|)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel) * torch.abs(asset.data.qfrc_actuator), dim=1
    )


def reward_dof_vel(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint velocities."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def reward_joint_tracking_error(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from joint position targets."""
    asset: Entity = env.scene[asset_cfg.name]
    target = env.action_manager.action + asset.data.default_joint_pos
    return torch.sum(torch.square(target - asset.data.joint_pos), dim=-1)


def reward_dof_pos_limits(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions beyond soft limits."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    lower = asset.data.soft_joint_pos_limits[..., 0]
    upper = asset.data.soft_joint_pos_limits[..., 1]
    out_of_limits = -(pos - lower).clip(max=0.0) + (pos - upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def reward_dof_vel_limits(
    env: "ManagerBasedRlEnv",
    soft_ratio: float = 0.9,
    default_vel_limit: float = 50.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint velocities beyond soft limits.

    Uses a fixed *default_vel_limit* since mjlab's EntityData does not expose
    per-joint velocity limits.  The penalty is clamped to [0, 1] per joint.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(
        (torch.abs(asset.data.joint_vel) - default_vel_limit * soft_ratio).clip(
            min=0.0, max=1.0
        ),
        dim=1,
    )


# ---------------------------------------------------------------------------
# Style Rewards
# ---------------------------------------------------------------------------


def reward_waist_deviation(
    env: "ManagerBasedRlEnv",
    threshold: float = 1.4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive waist rotation (waist_yaw_joint)."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = list(asset.joint_names)
    waist_idx = [i for i, n in enumerate(joint_names) if "waist_yaw" in n]
    if not waist_idx:
        return torch.zeros(env.num_envs, device=env.device)
    waist_pos = asset.data.joint_pos[:, waist_idx[0]]
    return (torch.abs(waist_pos) > threshold).float()


def reward_hip_yaw_deviation(
    env: "ManagerBasedRlEnv",
    max_threshold: float = 1.4,
    min_threshold: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive hip yaw."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = list(asset.joint_names)
    hip_yaw_idx = [i for i, n in enumerate(joint_names) if "hip_yaw" in n]
    if not hip_yaw_idx:
        return torch.zeros(env.num_envs, device=env.device)
    pos = asset.data.joint_pos[:, hip_yaw_idx]
    above_max = torch.max(torch.abs(pos), dim=-1)[0] > max_threshold
    above_min = torch.min(torch.abs(pos), dim=-1)[0] > min_threshold
    return (above_max | above_min).float()


def reward_hip_roll_deviation(
    env: "ManagerBasedRlEnv",
    max_threshold: float = 1.4,
    min_threshold: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive hip roll."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = list(asset.joint_names)
    hip_roll_idx = [i for i, n in enumerate(joint_names) if "hip_roll" in n]
    if not hip_roll_idx:
        return torch.zeros(env.num_envs, device=env.device)
    pos = asset.data.joint_pos[:, hip_roll_idx]
    above_max = torch.max(torch.abs(pos), dim=-1)[0] > max_threshold
    above_min = torch.min(torch.abs(pos), dim=-1)[0] > min_threshold
    return (above_max | above_min).float()


def reward_shoulder_roll_deviation(
    env: "ManagerBasedRlEnv",
    left_threshold: float = -0.02,
    right_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize shoulder roll beyond thresholds."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = list(asset.joint_names)
    left_idx = [i for i, n in enumerate(joint_names) if "left_shoulder_roll" in n]
    right_idx = [i for i, n in enumerate(joint_names) if "right_shoulder_roll" in n]
    if not left_idx or not right_idx:
        return torch.zeros(env.num_envs, device=env.device)
    reward = (asset.data.joint_pos[:, left_idx[0]] < left_threshold) | (
        asset.data.joint_pos[:, right_idx[0]] > right_threshold
    )
    return reward.float()


def reward_left_foot_displacement(
    env: "ManagerBasedRlEnv",
    sigma: float = -2.0,
    phase3_height: float = 0.65,
    foot_body_name: str = "left_ankle_pitch_link",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward left foot close to base when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    base_xy = asset.data.root_link_pos_w[:, :2]
    body_names = list(asset.body_names)
    foot_idx = body_names.index(foot_body_name)
    foot_xy = asset.data.body_link_pos_w[:, foot_idx, :2]
    foot_z = asset.data.body_link_pos_w[:, foot_idx, 2]
    mse = torch.sum(torch.square(base_xy - foot_xy), dim=-1).clamp(min=0.3)
    reward = torch.exp(mse * sigma) * (foot_z < 0.3).float()
    standup = (asset.data.root_link_pos_w[:, 2] > phase3_height).float()
    return reward * standup


def reward_right_foot_displacement(
    env: "ManagerBasedRlEnv",
    sigma: float = -2.0,
    phase3_height: float = 0.65,
    foot_body_name: str = "right_ankle_pitch_link",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward right foot close to base when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    base_xy = asset.data.root_link_pos_w[:, :2]
    body_names = list(asset.body_names)
    foot_idx = body_names.index(foot_body_name)
    foot_xy = asset.data.body_link_pos_w[:, foot_idx, :2]
    foot_z = asset.data.body_link_pos_w[:, foot_idx, 2]
    mse = torch.sum(torch.square(base_xy - foot_xy), dim=-1).clamp(min=0.3)
    reward = torch.exp(mse * sigma) * (foot_z < 0.3).float()
    standup = (asset.data.root_link_pos_w[:, 2] > phase3_height).float()
    return reward * standup


def reward_knee_deviation(
    env: "ManagerBasedRlEnv",
    max_threshold: float = 2.85,
    min_threshold: float = -0.06,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize knee positions out of range."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = list(asset.joint_names)
    knee_idx = [i for i, n in enumerate(joint_names) if "knee" in n]
    if not knee_idx:
        return torch.zeros(env.num_envs, device=env.device)
    pos = asset.data.joint_pos[:, knee_idx]
    reward = (torch.max(torch.abs(pos), dim=-1)[0] > max_threshold) | (
        torch.min(pos, dim=-1)[0] < min_threshold
    )
    return reward.float()


def reward_shank_orientation(
    env: "ManagerBasedRlEnv",
    phase1_height: float = 0.45,
    phase3_height: float = 0.65,
    post_task: bool = False,
    knee_body_names: list[str] | None = None,
    foot_body_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward vertical shank (knee-to-foot) orientation."""
    asset: Entity = env.scene[asset_cfg.name]
    body_names = list(asset.body_names)

    if knee_body_names is None:
        knee_body_names = ["left_knee_link", "right_knee_link"]
    if foot_body_names is None:
        foot_body_names = ["left_ankle_pitch_link", "right_ankle_pitch_link"]

    knee_indices = [body_names.index(n) for n in knee_body_names]
    foot_indices = [body_names.index(n) for n in foot_body_names]

    knee_pos = asset.data.body_link_pos_w[:, knee_indices, :3]
    foot_pos = asset.data.body_link_pos_w[:, foot_indices, :3]

    diff = knee_pos - foot_pos
    orientation_z = diff[..., 2] / torch.norm(diff, dim=-1).clamp(min=1e-6)
    feet_orientation = orientation_z.mean(dim=-1)

    base_height = (asset.data.root_link_pos_w[:, 2] > phase1_height).float()
    reward = _tolerance(feet_orientation, (0.8, np.inf), 1.0, 0.1) * base_height

    if post_task:
        standup = asset.data.root_link_pos_w[:, 2] > phase3_height
        reward = reward * (~standup).float() + standup.float()
    return reward


def reward_ground_parallel(
    env: "ManagerBasedRlEnv",
    var_threshold: float = 0.05,
    phase3_height: float = 0.65,
    post_task: bool = False,
    left_ankle_body_names: list[str] | None = None,
    right_ankle_body_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward feet parallel to ground (low variance in ankle z height)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_names = list(asset.body_names)

    if left_ankle_body_names is None:
        left_ankle_body_names = ["left_ankle_roll_link"]
    if right_ankle_body_names is None:
        right_ankle_body_names = ["right_ankle_roll_link"]

    left_idx = [body_names.index(n) for n in left_ankle_body_names]
    right_idx = [body_names.index(n) for n in right_ankle_body_names]

    left_z = asset.data.body_link_pos_w[:, left_idx, 2] * 10
    right_z = asset.data.body_link_pos_w[:, right_idx, 2] * 10

    left_var = (
        left_z.var(dim=1) if left_z.shape[1] > 1
        else torch.zeros(env.num_envs, device=env.device)
    )
    right_var = (
        right_z.var(dim=1) if right_z.shape[1] > 1
        else torch.zeros(env.num_envs, device=env.device)
    )
    var = (left_var + right_var) / 2.0
    reward = (var < var_threshold).float()

    if post_task:
        standup = asset.data.root_link_pos_w[:, 2] > phase3_height
        reward = reward * (~standup).float() + standup.float()
    return reward


def reward_feet_distance(
    env: "ManagerBasedRlEnv",
    max_distance: float = 0.9,
    left_foot_body: str = "left_ankle_pitch_link",
    right_foot_body: str = "right_ankle_pitch_link",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet too far apart."""
    asset: Entity = env.scene[asset_cfg.name]
    body_names = list(asset.body_names)
    left_idx = body_names.index(left_foot_body)
    right_idx = body_names.index(right_foot_body)
    left_pos = asset.data.body_link_pos_w[:, left_idx, :3]
    right_pos = asset.data.body_link_pos_w[:, right_idx, :3]
    dist = torch.norm(left_pos - right_pos, dim=-1)
    return (dist > max_distance).float()


def reward_style_ang_vel_xy(
    env: "ManagerBasedRlEnv",
    phase1_height: float = 0.45,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage low angular velocity when above phase1 height."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_link_ang_vel_b[:, :2]
    base_height = (asset.data.root_link_pos_w[:, 2] > phase1_height).float()
    return torch.exp(torch.sum(torch.square(ang_vel), dim=1) * -2) * base_height


# ---------------------------------------------------------------------------
# Post-Task (Target) Rewards
# ---------------------------------------------------------------------------


def reward_target_ang_vel_xy(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage low angular velocity when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_link_ang_vel_b[:, :2]
    standup = (asset.data.root_link_pos_w[:, 2] > phase3_height).float()
    return torch.exp(torch.sum(torch.square(ang_vel), dim=1) * -2) * standup


def reward_target_lin_vel_xy(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage low linear velocity when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_link_lin_vel_b[:, :2]
    standup = (asset.data.root_link_pos_w[:, 2] > phase3_height).float()
    return torch.exp(torch.sum(torch.square(lin_vel), dim=1) * -5) * standup


def reward_feet_height_var(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    left_foot_body: str = "left_ankle_pitch_link",
    right_foot_body: str = "right_ankle_pitch_link",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage equal feet height when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    body_names = list(asset.body_names)
    left_idx = body_names.index(left_foot_body)
    right_idx = body_names.index(right_foot_body)
    left_h = asset.data.body_link_pos_w[:, left_idx, 2] * 10
    right_h = asset.data.body_link_pos_w[:, right_idx, 2] * 10
    diff = torch.abs(left_h - right_h).clamp(min=0.2)
    standup = (asset.data.root_link_pos_w[:, 2] > phase3_height).float()
    return torch.exp(diff * -2) * standup


def reward_target_upper_dof_pos(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    sigma: float = -0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage upper body to reach target pose when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = list(asset.joint_names)
    upper_idx = [
        i
        for i, n in enumerate(joint_names)
        if any(k in n for k in ["shoulder", "elbow", "wrist"])
    ]
    if not upper_idx:
        return torch.zeros(env.num_envs, device=env.device)
    target = asset.data.default_joint_pos[:, upper_idx]
    current = asset.data.joint_pos[:, upper_idx]
    mse = torch.sum(torch.square(current - target), dim=-1)
    standup = (asset.data.root_link_pos_w[:, 2] > phase3_height).float()
    return torch.exp(mse * sigma) * standup


def reward_target_orientation(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage flat orientation when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity_proj = asset.data.projected_gravity_b[:, :2]
    standup = (asset.data.root_link_pos_w[:, 2] > phase3_height).float()
    return torch.exp(torch.sum(torch.square(gravity_proj), dim=1) * -5) * standup


def reward_target_base_height(
    env: "ManagerBasedRlEnv",
    target_height: float = 0.75,
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage target base height when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    standup = (base_height > phase3_height).float()
    return torch.exp(torch.abs(base_height - target_height) * -20) * standup
