"""Custom event functions for HoST stand-up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_joints_scaled(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] = (0.9, 1.1),
    offset_range: tuple[float, float] = (-0.05, 0.05),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*_joint",)),
) -> None:
    """Reset around the configured pose using multiplicative and additive noise."""
    reset_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids_tensor = torch.tensor(joint_ids, device=env.device)
    else:
        joint_ids_tensor = joint_ids

    default_pos = asset.data.default_joint_pos[reset_ids][:, joint_ids].clone()
    scale = torch.empty_like(default_pos).uniform_(*scale_range)
    offset = torch.empty_like(default_pos).uniform_(*offset_range)
    joint_pos = default_pos * scale + offset
    limits = asset.data.soft_joint_pos_limits[reset_ids][:, joint_ids]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])

    default_vel = asset.data.default_joint_vel[reset_ids][:, joint_ids].clone()
    joint_vel = default_vel + torch.empty_like(default_vel).uniform_(*velocity_range)
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=reset_ids,
        joint_ids=joint_ids_tensor,
    )


def apply_traction_force(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    initial_force: float = 100.0,
    no_orientation: bool = False,
    unactuated_steps: int = 30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply upward traction force to the robot's root body (pelvis).

    Called as an interval event during simulation. Force magnitude comes from
    the curriculum buffer ``host_traction_force``. The force is applied in the
    world frame (+Z direction) to the root body (body index 0).

    Args:
        env: The manager-based RL environment.
        env_ids: Environment indices to apply forces to.
        initial_force: Initial force magnitude if buffer doesn't exist yet.
        no_orientation: If True, skip orientation gating.
        unactuated_steps: Number of steps to wait before applying force.
        asset_cfg: Scene entity configuration for the robot.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Initialize force buffer on first call
    if not hasattr(env, "host_traction_force"):
        env.host_traction_force = torch.full(
            (env.num_envs,), initial_force, device=env.device
        )

    # Only apply after unactuated period
    active_mask = env.episode_length_buf > unactuated_steps

    # Optionally gate on orientation (robot must be somewhat upright)
    if not no_orientation:
        gravity_proj = asset.data.projected_gravity_b
        orientation_mask = gravity_proj[:, 2] < -0.8
        active_mask = active_mask & orientation_mask

    # Build force tensor: upward force on root body (index 0)
    forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    forces[:, 0, 2] = env.host_traction_force * active_mask.float()

    asset.write_external_wrench_to_sim(
        forces,
        torch.zeros_like(forces),
        env_ids=env_ids,
        body_ids=[0],
    )


def reset_action_rescale(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    initial_rescale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Initialize action rescale buffer on reset if not already done."""
    if not hasattr(env, "host_action_rescale"):
        env.host_action_rescale = torch.full(
            (env.num_envs,), initial_rescale, device=env.device
        )


def update_last_last_action(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Update the last-last action buffer after each step for smoothness reward."""
    num_actions = env.action_manager.action.shape[1]
    if not hasattr(env, "host_last_last_action"):
        env.host_last_last_action = torch.zeros(
            env.num_envs, num_actions, device=env.device
        )
    env.host_last_last_action[env_ids] = env.action_manager.prev_action[env_ids].clone()


def reset_last_last_action(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the last-last action buffer used for smoothness reward on reset."""
    num_actions = env.action_manager.action.shape[1]
    if not hasattr(env, "host_last_last_action"):
        env.host_last_last_action = torch.zeros(
            env.num_envs, num_actions, device=env.device
        )
    env.host_last_last_action[env_ids] = 0.0


def reset_standup_success(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    initial_hold_steps: int = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the per-episode stand-up success latch."""
    if not hasattr(env, "host_standup_success"):
        env.host_standup_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "host_post_task_steps"):
        env.host_post_task_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_candidate_steps"):
        env.host_standup_candidate_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_lost_steps"):
        env.host_standup_lost_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_hold_steps"):
        env.host_standup_hold_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_hold_reached"):
        env.host_standup_hold_reached = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    # This curriculum target persists across episode resets.  The elapsed
    # counter and completion latch are reset below for each new episode.
    if not hasattr(env, "host_standup_hold_required_steps"):
        env.host_standup_hold_required_steps = torch.full(
            (env.num_envs,), initial_hold_steps, dtype=torch.long, device=env.device
        )
    asset: Entity = env.scene[asset_cfg.name]
    num_joints = asset.data.joint_pos.shape[1]
    num_actions = env.action_manager.action.shape[1]
    if not hasattr(env, "host_post_pose_locked"):
        env.host_post_pose_locked = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "host_joint_pose_ref"):
        env.host_joint_pose_ref = torch.zeros(
            env.num_envs, num_joints, device=env.device
        )
    if not hasattr(env, "host_action_ref"):
        env.host_action_ref = torch.zeros(
            env.num_envs, num_actions, device=env.device
        )
    env.host_standup_success[env_ids] = False
    env.host_post_task_steps[env_ids] = 0
    env.host_standup_candidate_steps[env_ids] = 0
    env.host_standup_lost_steps[env_ids] = 0
    env.host_standup_hold_steps[env_ids] = 0
    env.host_standup_hold_reached[env_ids] = False
    env.host_post_pose_locked[env_ids] = False
    env.host_joint_pose_ref[env_ids] = 0.0
    env.host_action_ref[env_ids] = 0.0


def update_standup_success(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    threshold_height: float = 0.65,
    upright_gravity_z: float = -0.8,
    min_torso_height: float = 0.70,
    candidate_steps: int = 1,
    hold_steps: int = 5,
    hold_threshold_height: float = 0.60,
    hold_min_torso_height: float = 0.66,
    hold_upright_gravity_z: float = -0.70,
    hold_threshold_height_initial: float = 0.52,
    hold_min_torso_height_initial: float = 0.45,
    hold_upright_gravity_z_initial: float = -0.55,
    hold_settle_steps: int = 35,
    quiet_root_ang_vel: float = 1.0,
    quiet_root_lin_vel: float = 0.6,
    quiet_joint_vel: float = 2.0,
    quiet_root_ang_vel_initial: float = 3.0,
    quiet_root_lin_vel_initial: float = 2.0,
    quiet_joint_vel_initial: float = 8.0,
    no_orientation: bool = False,
    torso_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Latch the upright phase on a strict standing frame and track the hold."""
    if not hasattr(env, "host_standup_success"):
        env.host_standup_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "host_post_task_steps"):
        env.host_post_task_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_candidate_steps"):
        env.host_standup_candidate_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_lost_steps"):
        env.host_standup_lost_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_hold_steps"):
        env.host_standup_hold_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_hold_reached"):
        env.host_standup_hold_reached = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "host_standup_hold_required_steps"):
        env.host_standup_hold_required_steps = torch.full(
            (env.num_envs,), hold_steps, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_hold_settle_steps"):
        env.host_standup_hold_settle_steps = torch.full(
            (env.num_envs,), hold_settle_steps, dtype=torch.long, device=env.device
        )
    asset: Entity = env.scene[asset_cfg.name]
    num_joints = asset.data.joint_pos.shape[1]
    num_actions = env.action_manager.action.shape[1]
    if not hasattr(env, "host_post_pose_locked"):
        env.host_post_pose_locked = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "host_joint_pose_ref"):
        env.host_joint_pose_ref = torch.zeros(
            env.num_envs, num_joints, device=env.device
        )
    if not hasattr(env, "host_action_ref"):
        env.host_action_ref = torch.zeros(
            env.num_envs, num_actions, device=env.device
        )
    torso_idx = list(asset.body_names).index(torso_body_name)
    standing = (
        (asset.data.root_link_pos_w[env_ids, 2] > threshold_height)
        & (asset.data.body_link_pos_w[env_ids, torso_idx, 2] > min_torso_height)
    )
    if not no_orientation:
        standing &= asset.data.projected_gravity_b[env_ids, 2] < upright_gravity_z

    # The hold gate has a small hysteresis band. Contact settling and sensor
    # noise should not erase an otherwise valid standing window.
    hold_standing = (
        (asset.data.root_link_pos_w[env_ids, 2] > hold_threshold_height)
        & (asset.data.body_link_pos_w[env_ids, torso_idx, 2] > hold_min_torso_height)
    )
    if not no_orientation:
        hold_standing &= (
            asset.data.projected_gravity_b[env_ids, 2] < hold_upright_gravity_z
        )
    # A curriculum hold is only valid after the post-task settle window and
    # requires genuinely low motion. Height alone would let a briefly upright
    # but still oscillating robot advance the assistance course.
    settle_targets = getattr(env, "host_standup_hold_settle_steps", None)
    if settle_targets is None:
        settle_targets = torch.full(
            (env.num_envs,), hold_settle_steps, dtype=torch.long, device=env.device
        )
    hold_standing &= env.host_post_task_steps[env_ids] >= (
        settle_targets[env_ids] - 1
    ).clamp(min=0)
    course_level = getattr(env, "host_curriculum_level", None)
    if course_level is None:
        course_level = torch.zeros((), device=env.device)
    course_level = course_level.clamp(0.0, 1.0)
    current_hold_threshold_height = hold_threshold_height_initial + (
        hold_threshold_height - hold_threshold_height_initial
    ) * course_level
    current_hold_min_torso_height = hold_min_torso_height_initial + (
        hold_min_torso_height - hold_min_torso_height_initial
    ) * course_level
    current_hold_upright_gravity_z = hold_upright_gravity_z_initial + (
        hold_upright_gravity_z - hold_upright_gravity_z_initial
    ) * course_level
    hold_standing = (
        (asset.data.root_link_pos_w[env_ids, 2] > current_hold_threshold_height)
        & (
            asset.data.body_link_pos_w[env_ids, torso_idx, 2]
            > current_hold_min_torso_height
        )
    )
    if not no_orientation:
        hold_standing &= (
            asset.data.projected_gravity_b[env_ids, 2]
            < current_hold_upright_gravity_z
        )
    current_quiet_root_ang_vel = quiet_root_ang_vel_initial + (
        quiet_root_ang_vel - quiet_root_ang_vel_initial
    ) * course_level
    current_quiet_root_lin_vel = quiet_root_lin_vel_initial + (
        quiet_root_lin_vel - quiet_root_lin_vel_initial
    ) * course_level
    current_quiet_joint_vel = quiet_joint_vel_initial + (
        quiet_joint_vel - quiet_joint_vel_initial
    ) * course_level
    hold_standing &= (
        torch.linalg.vector_norm(
            asset.data.root_link_ang_vel_b[env_ids, :2], dim=1
        )
        < current_quiet_root_ang_vel
    )
    hold_standing &= (
        torch.linalg.vector_norm(
            asset.data.root_link_lin_vel_b[env_ids, :2], dim=1
        )
        < current_quiet_root_lin_vel
    )
    hold_standing &= (
        torch.mean(torch.abs(asset.data.joint_vel[env_ids]), dim=1)
        < current_quiet_joint_vel
    )
    env.host_standup_candidate_steps[env_ids] = torch.where(
        standing,
        env.host_standup_candidate_steps[env_ids] + 1,
        torch.zeros_like(env.host_standup_candidate_steps[env_ids]),
    )
    confirmed = env.host_standup_candidate_steps[env_ids] >= candidate_steps
    env.host_standup_success[env_ids] |= confirmed
    active = env.host_standup_success[env_ids]
    # This is deliberately separate from the elapsed post-task counter:
    # only consecutive, currently-standing frames can complete the dynamic
    # hold window used by the curriculum.
    env.host_standup_hold_steps[env_ids] = torch.where(
        active & hold_standing,
        env.host_standup_hold_steps[env_ids] + 1,
        torch.zeros_like(env.host_standup_hold_steps[env_ids]),
    )
    required_hold_steps = env.host_standup_hold_required_steps[env_ids]
    hold_reached_now = (
        env.host_standup_hold_steps[env_ids] >= required_hold_steps
    )
    env.host_standup_hold_reached[env_ids] |= hold_reached_now
    env.host_post_task_steps[env_ids] = torch.where(
        active,
        env.host_post_task_steps[env_ids] + 1,
        torch.zeros_like(env.host_post_task_steps[env_ids]),
    )
    # Let the final contact adjustments settle, then lock the naturally
    # emerging joint/action state.  This is a per-episode reference, not a
    # hand-written pose or world-frame orientation target.
    # Let the complete stand-up motion settle before freezing the natural
    # pose.  Posture shaping starts at 20 policy steps and therefore has a
    # 15-step window to improve symmetry before this reference is captured.
    pose_lock_delay_steps = 35
    lock_now = (
        active
        & ~env.host_post_pose_locked[env_ids]
        & (env.host_post_task_steps[env_ids] >= pose_lock_delay_steps)
    )
    env.host_joint_pose_ref[env_ids] = torch.where(
        lock_now[:, None], asset.data.joint_pos[env_ids], env.host_joint_pose_ref[env_ids]
    )
    env.host_action_ref[env_ids] = torch.where(
        lock_now[:, None],
        env.action_manager.action[env_ids],
        env.host_action_ref[env_ids],
    )
    env.host_post_pose_locked[env_ids] |= lock_now
