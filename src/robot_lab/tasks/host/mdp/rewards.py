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
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

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


def _post_task_gate(
    asset: Entity,
    phase3_height: float,
    upright_gravity_z: float = -0.70,
) -> torch.Tensor:
    """Gate post-task rewards on both standing height and upright attitude."""
    return (
        (asset.data.root_link_pos_w[:, 2] > phase3_height)
        & (asset.data.projected_gravity_b[:, 2] < upright_gravity_z)
    ).float()


def _post_task_settled(
    env: "ManagerBasedRlEnv",
    asset: Entity,
    phase3_height: float,
    settle_steps: int = 35,
) -> torch.Tensor:
    """Post-task gate delayed until the upright state has settled briefly."""
    course_level = getattr(env, "host_curriculum_level", None)
    if course_level is None:
        course_level = torch.zeros((), device=env.device)
    course_level = course_level.clamp(0.0, 1.0)
    # Early levels shape quiet standing as soon as the torso is nearly upright;
    # the gate rises linearly to the final geometric standing target.
    early_height = min(0.50, phase3_height)
    gate_height = early_height + (phase3_height - early_height) * course_level
    gate_upright = -0.45 + (-0.80 + 0.45) * course_level
    gate = (
        (asset.data.root_link_pos_w[:, 2] > gate_height)
        & (asset.data.projected_gravity_b[:, 2] < gate_upright)
    ).float()
    active = getattr(env, "host_standup_success", None)
    if active is not None:
        gate = gate * active.float()
    steps = getattr(env, "host_post_task_steps", None)
    if steps is None:
        return gate
    # The same outer curriculum that controls the hold counter also controls
    # when static post-task shaping starts.  Otherwise early large-action
    # episodes receive no quiet-standing signal until the final 35-step delay.
    settle_targets = getattr(env, "host_standup_hold_settle_steps", None)
    if settle_targets is None:
        settle_targets = torch.full(
            (env.num_envs,), settle_steps, dtype=torch.long, device=env.device
        )
    # Keep a short post-latch buffer even at the easiest course level so the
    # post critic cannot suppress the large final stand-up correction.
    settle_targets = torch.maximum(
        settle_targets,
        torch.full_like(settle_targets, min(10, settle_steps)),
    )
    return gate * (steps >= settle_targets).float()


def _post_pose_lock(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return the per-episode natural-pose lock mask."""
    locked = getattr(env, "host_post_pose_locked", None)
    if locked is None:
        return torch.zeros(env.num_envs, device=env.device)
    return locked.float()


# ---------------------------------------------------------------------------
# Task Rewards
# ---------------------------------------------------------------------------


def reward_orientation(
    env: "ManagerBasedRlEnv",
    phase1_height: float = 0.45,
    orientation_threshold: float = 0.99,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Densely reward upright orientation throughout the stand-up motion."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity_proj = asset.data.projected_gravity_b
    reward = _tolerance(
        -gravity_proj[:, 2], (orientation_threshold, np.inf), 1.0, 0.05
    )
    return reward


def reward_head_height(
    env: "ManagerBasedRlEnv",
    target_head_height: float = 1.0,
    target_head_margin: float = 1.0,
    head_body_name: str = "torso_link",
    foot_body_names: list[str] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward torso/head height in the world frame, as defined in HoST."""
    asset: Entity = env.scene[asset_cfg.name]
    body_names = list(asset.body_names)

    head_idx = body_names.index(head_body_name)
    head_height = asset.data.body_link_pos_w[:, head_idx, 2:3]

    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_indices = [body_names.index(n) for n in foot_body_names]
    reward = _tolerance(
        head_height, (target_head_height, np.inf), target_head_margin, 0.1
    )
    return reward.squeeze(-1)


def reward_standup_hold_progress(
    env: "ManagerBasedRlEnv",
    threshold_height: float = 0.60,
    min_torso_height: float = 0.66,
    upright_gravity_z: float = -0.70,
    settle_steps: int = 35,
    quiet_root_ang_vel: float = 1.0,
    quiet_root_lin_vel: float = 0.6,
    quiet_joint_vel: float = 2.0,
    quiet_root_ang_vel_initial: float = 3.0,
    quiet_root_lin_vel_initial: float = 2.0,
    quiet_joint_vel_initial: float = 8.0,
    hold_threshold_height_initial: float = 0.52,
    hold_min_torso_height_initial: float = 0.45,
    hold_upright_gravity_z_initial: float = -0.55,
    no_orientation: bool = False,
    torso_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward quiet standing with a dense, yaw-invariant soft gate.

    ``host_standup_success`` is still the phase latch, but the geometric and
    velocity checks are deliberately continuous.  The boolean checks in the
    event function remain the curriculum/diagnostic definition; they are too
    sparse to be the only policy-learning signal when the robot is settling.
    """
    active = getattr(env, "host_standup_success", None)
    hold_steps = getattr(env, "host_standup_hold_steps", None)
    required = getattr(env, "host_standup_hold_required_steps", None)
    if active is None or hold_steps is None or required is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    torso_idx = list(asset.body_names).index(torso_body_name)
    root_height = asset.data.root_link_pos_w[:, 2]
    torso_height = asset.data.body_link_pos_w[:, torso_idx, 2]
    post_steps = getattr(env, "host_post_task_steps", None)
    if post_steps is not None:
        settle_targets = getattr(env, "host_standup_hold_settle_steps", None)
        if settle_targets is None:
            settle_targets = torch.full(
                (env.num_envs,), settle_steps, dtype=torch.long, device=env.device
            )
        settle_progress = (
            post_steps.to(dtype=torch.float32)
            / settle_targets.to(dtype=torch.float32).clamp(min=1.0)
        ).clamp(0.0, 1.0)
    else:
        settle_progress = torch.ones(env.num_envs, device=env.device)
    course_level = getattr(env, "host_curriculum_level", None)
    if course_level is None:
        course_level = torch.zeros((), device=env.device)
    course_level = course_level.clamp(0.0, 1.0)
    current_hold_threshold_height = hold_threshold_height_initial + (
        threshold_height - hold_threshold_height_initial
    ) * course_level
    current_hold_min_torso_height = hold_min_torso_height_initial + (
        min_torso_height - hold_min_torso_height_initial
    ) * course_level
    current_hold_upright_gravity_z = hold_upright_gravity_z_initial + (
        upright_gravity_z - hold_upright_gravity_z_initial
    ) * course_level
    height_gate = _tolerance(
        root_height,
        (current_hold_threshold_height, np.inf),
        margin=0.10,
        value_at_margin=0.10,
    )
    height_gate = height_gate * _tolerance(
        torso_height,
        (current_hold_min_torso_height, np.inf),
        margin=0.12,
        value_at_margin=0.10,
    )
    if no_orientation:
        upright_gate = torch.ones_like(height_gate)
    else:
        gravity_xy = asset.data.projected_gravity_b[:, :2]
        # Projected gravity is invariant to world yaw, while roll/pitch
        # excursions are still strongly discouraged.
        upright_gate = torch.exp(-6.0 * torch.sum(torch.square(gravity_xy), dim=1))
    current_quiet_root_ang_vel = quiet_root_ang_vel_initial + (
        quiet_root_ang_vel - quiet_root_ang_vel_initial
    ) * course_level
    current_quiet_root_lin_vel = quiet_root_lin_vel_initial + (
        quiet_root_lin_vel - quiet_root_lin_vel_initial
    ) * course_level
    current_quiet_joint_vel = quiet_joint_vel_initial + (
        quiet_joint_vel - quiet_joint_vel_initial
    ) * course_level
    ang_xy = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b[:, :2], dim=1)
    lin_xy = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    joint_rms = torch.sqrt(torch.mean(torch.square(asset.data.joint_vel), dim=1))
    quiet_gate = torch.exp(
        -torch.square(ang_xy / current_quiet_root_ang_vel.clamp(min=0.1))
        -torch.square(lin_xy / current_quiet_root_lin_vel.clamp(min=0.1))
        -torch.square(joint_rms / current_quiet_joint_vel.clamp(min=0.1))
    )
    progress = (
        (hold_steps.to(dtype=torch.float32) + 1.0)
        / required.to(dtype=torch.float32).clamp(min=1.0)
    ).clamp(0.0, 1.0)
    return active.float() * progress * height_gate * upright_gate * quiet_gate * settle_progress


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
    action_term = env.action_manager.get_term("joint_pos")
    target = action_term.target_joint_pos
    current = asset.data.joint_pos[:, action_term.target_ids]
    return torch.sum(torch.square(target - current), dim=-1)


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
    standup = _post_task_gate(asset, phase3_height)
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
    standup = _post_task_gate(asset, phase3_height)
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

    return reward


def reward_ground_parallel(
    env: "ManagerBasedRlEnv",
    var_threshold: float = 0.05,
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
    ang_vel = asset.data.root_link_ang_vel_b
    # Yaw is free in angle, but sustained yaw spin is still undesirable.
    weighted_sq = (
        torch.sum(torch.square(ang_vel[:, :2]), dim=1)
        + 0.25 * torch.square(ang_vel[:, 2])
    )
    base_height = (asset.data.root_link_pos_w[:, 2] > phase1_height).float()
    return torch.exp(weighted_sq * -2) * base_height


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
    ang_vel = asset.data.root_link_ang_vel_b
    weighted_sq = (
        torch.sum(torch.square(ang_vel[:, :2]), dim=1)
        + 0.25 * torch.square(ang_vel[:, 2])
    )
    standup = _post_task_gate(asset, phase3_height)
    return torch.exp(weighted_sq * -2) * standup


def reward_post_root_ang_vel_l2(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task cost for roll/pitch motion and weaker yaw spin."""
    asset: Entity = env.scene[asset_cfg.name]
    standup = _post_task_settled(env, asset, phase3_height)
    ang_vel = asset.data.root_link_ang_vel_b
    weighted_sq = (
        torch.sum(torch.square(ang_vel[:, :2]), dim=1)
        + 0.25 * torch.square(ang_vel[:, 2])
    )
    return weighted_sq * standup


def reward_target_lin_vel_xy(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage low linear velocity when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_link_lin_vel_b[:, :2]
    standup = _post_task_gate(asset, phase3_height)
    return torch.exp(torch.sum(torch.square(lin_vel), dim=1) * -5) * standup


def reward_post_root_lin_vel_l2(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task cost for all three root linear-velocity components."""
    asset: Entity = env.scene[asset_cfg.name]
    standup = _post_task_settled(env, asset, phase3_height)
    return torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1) * standup


def reward_post_joint_vel_l2(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task cost for residual joint motion."""
    asset: Entity = env.scene[asset_cfg.name]
    standup = _post_task_settled(env, asset, phase3_height)
    return torch.sum(torch.square(asset.data.joint_vel), dim=1) * standup


def reward_post_joint_acc_l2(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task cost for high-frequency joint jitter."""
    asset: Entity = env.scene[asset_cfg.name]
    standup = _post_task_settled(env, asset, phase3_height)
    return torch.sum(torch.square(asset.data.joint_acc), dim=1) * standup


def reward_post_action_rate_l2(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task cost for changing the joint targets."""
    asset: Entity = env.scene[asset_cfg.name]
    standup = _post_task_settled(env, asset, phase3_height)
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1) * standup


def reward_post_smoothness_l2(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task cost for second-order target changes."""
    asset: Entity = env.scene[asset_cfg.name]
    standup = _post_task_settled(env, asset, phase3_height)
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    if hasattr(env, "host_last_last_action"):
        delta2 = action - 2 * prev + env.host_last_last_action
    else:
        delta2 = action - prev
    return torch.sum(torch.square(delta2), dim=1) * standup


def reward_post_pose_drift(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize drifting from the naturally captured standing joint pose."""
    asset: Entity = env.scene[asset_cfg.name]
    reference = getattr(env, "host_joint_pose_ref", None)
    if reference is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.mean(torch.square(asset.data.joint_pos - reference), dim=1) * _post_pose_lock(env)


def reward_post_action_drift(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize changing the target action after the natural pose is locked."""
    reference = getattr(env, "host_action_ref", None)
    if reference is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.mean(
        torch.square(env.action_manager.action - reference), dim=1
    ) * _post_pose_lock(env)


_POST_SYMMETRIC_JOINTS = (
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

# Sign of the mirrored joint displacement around the configured default pose.
_POST_MIRROR_PAIRS = (
    ("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
    ("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
    ("left_hip_yaw_joint", "right_hip_yaw_joint", 1.0),
    ("left_knee_joint", "right_knee_joint", 1.0),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1.0),
    ("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0),
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1.0),
    ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1.0),
    ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", 1.0),
    ("left_elbow_joint", "right_elbow_joint", 1.0),
    ("left_wrist_roll_joint", "right_wrist_roll_joint", -1.0),
)


def reward_post_symmetric_pose(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    settle_steps: int = 50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize one concise, mirrored full-body standing pose.

    The configured G1 default pose is the reference: it places both arms
    beside the waist, keeps the waist yaw at zero, and has mirrored legs.
    Pair terms additionally prevent left/right drift away from that symmetry.
    No hand Cartesian target or world-frame heading is used.
    """
    asset: Entity = env.scene[asset_cfg.name]
    names = list(asset.joint_names)
    joint_ids = [names.index(name) for name in _POST_SYMMETRIC_JOINTS if name in names]
    if not joint_ids:
        return torch.zeros(env.num_envs, device=env.device)

    current = asset.data.joint_pos[:, joint_ids]
    reference = asset.data.default_joint_pos[:, joint_ids]
    # A 0.25 rad scale makes a visibly wrong joint contribute an O(1) cost;
    # waist yaw uses a tighter scale because it is explicitly neutral.
    scale = torch.as_tensor(
        [0.12 if names[joint_id] == "waist_yaw_joint" else 0.25 for joint_id in joint_ids],
        device=env.device,
        dtype=current.dtype,
    )
    target_cost = torch.mean(torch.square((current - reference) / scale), dim=1)

    pair_costs: list[torch.Tensor] = []
    for left_name, right_name, mirror_sign in _POST_MIRROR_PAIRS:
        if left_name not in names or right_name not in names:
            continue
        left = asset.data.joint_pos[:, names.index(left_name)]
        right = asset.data.joint_pos[:, names.index(right_name)]
        left_ref = asset.data.default_joint_pos[:, names.index(left_name)]
        right_ref = asset.data.default_joint_pos[:, names.index(right_name)]
        pair_costs.append(
            torch.square(((left - left_ref) - mirror_sign * (right - right_ref)) / 0.25)
        )
    symmetry_cost = (
        torch.stack(pair_costs, dim=1).mean(dim=1)
        if pair_costs
        else torch.zeros_like(target_cost)
    )
    standup = _post_task_settled(env, asset, phase3_height, settle_steps)
    return (target_cost + 0.5 * symmetry_cost) * standup


def reward_post_foot_alignment(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    target_width: float = 0.237,
    target_x: float = 0.0,
    settle_steps: int = 50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize wide, staggered, or non-parallel feet in the root frame."""
    asset: Entity = env.scene[asset_cfg.name]
    body_names = list(asset.body_names)
    foot_ids = torch.as_tensor(
        [
            body_names.index("left_ankle_roll_link"),
            body_names.index("right_ankle_roll_link"),
        ],
        device=env.device,
        dtype=torch.long,
    )
    root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    foot_delta_w = (
        asset.data.body_link_pos_w[:, foot_ids, :]
        - asset.data.root_link_pos_w[:, None, :]
    )
    foot_pos_b = quat_apply_inverse(root_quat, foot_delta_w)
    width = foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1]
    position_cost = (
        torch.square((width - target_width) / 0.04)
        + torch.square((foot_pos_b[:, 0, 0] - foot_pos_b[:, 1, 0]) / 0.04)
        + torch.square(foot_pos_b[:, :, 1].mean(dim=1) / 0.03)
        + torch.square((foot_pos_b[:, 0, 2] - foot_pos_b[:, 1, 2]) / 0.03)
    )

    forward_w = torch.zeros(
        (env.num_envs, 2, 3), device=env.device, dtype=foot_delta_w.dtype
    )
    forward_w[..., 0] = 1.0
    foot_forward_w = quat_apply(asset.data.body_link_quat_w[:, foot_ids, :], forward_w)
    foot_forward_b = quat_apply_inverse(root_quat, foot_forward_w)
    parallel_cost = torch.sum(torch.square(foot_forward_b[:, 0] - foot_forward_b[:, 1]), dim=1)
    standup = _post_task_settled(env, asset, phase3_height, settle_steps)
    return (position_cost + parallel_cost) * standup


def _foot_contact_by_side(
    contact_sensor: ContactSensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-environment left/right contact masks from foot geoms."""
    found = contact_sensor.data.found
    if found is None:
        raise RuntimeError("feet_ground_contact must provide the 'found' field")
    if found.ndim == 3:
        found = found.squeeze(-1)
    names = [str(name).lower() for name in contact_sensor.primary_names]
    left_ids = [i for i, name in enumerate(names) if "left_foot" in name]
    right_ids = [i for i, name in enumerate(names) if "right_foot" in name]
    if not left_ids or not right_ids:
        any_contact = (found > 0).any(dim=1)
        return any_contact, any_contact
    left = (found[:, left_ids] > 0).any(dim=1)
    right = (found[:, right_ids] > 0).any(dim=1)
    return left, right


def reward_post_feet_slip(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    sensor_name: str = "feet_ground_contact",
    left_foot_body: str = "left_ankle_roll_link",
    right_foot_body: str = "right_ankle_roll_link",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task cost for horizontal velocity of feet that are in contact."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    body_names = list(asset.body_names)
    foot_ids = [body_names.index(left_foot_body), body_names.index(right_foot_body)]
    foot_vel_xy = asset.data.body_link_lin_vel_w[:, foot_ids, :2]
    left_contact, right_contact = _foot_contact_by_side(sensor)
    contact = torch.stack((left_contact, right_contact), dim=1).to(foot_vel_xy.dtype)
    cost = torch.sum(torch.square(foot_vel_xy).sum(dim=-1) * contact, dim=1)
    standup = _post_task_settled(env, asset, phase3_height)
    return cost * standup


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
    standup = _post_task_gate(asset, phase3_height)
    return torch.exp(mse * sigma) * standup


def reward_target_orientation(
    env: "ManagerBasedRlEnv",
    phase3_height: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Encourage flat orientation when standing."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity_proj = asset.data.projected_gravity_b[:, :2]
    standup = _post_task_gate(asset, phase3_height)
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
    standup = _post_task_gate(asset, phase3_height)
    return torch.exp(torch.square(base_height - target_height) * -20) * standup
