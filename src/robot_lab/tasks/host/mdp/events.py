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


STANDUP_PHASE = 0
TRANSITION_PHASE = 1
POST_TASK_PHASE = 2


def reset_standup_success(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
) -> None:
    """Reset the realtime stand-up phase and episode outcome state."""
    if not hasattr(env, "host_standup_phase"):
        env.host_standup_phase = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_reached"):
        env.host_standup_reached = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "host_standup_transition_steps"):
        env.host_standup_transition_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_post_task_steps"):
        env.host_post_task_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_candidate_steps"):
        env.host_standup_candidate_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    env.host_standup_phase[env_ids] = STANDUP_PHASE
    env.host_standup_reached[env_ids] = False
    env.host_standup_candidate_steps[env_ids] = 0
    env.host_standup_transition_steps[env_ids] = 0
    env.host_post_task_steps[env_ids] = 0


def update_standup_success(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    upright_gravity_z: float = -0.55,
    min_head_height: float = 1.27,
    candidate_steps: int = 3,
    transition_steps_required: int = 10,
    no_orientation: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Advance the realtime STANDUP -> TRANSITION -> POST_TASK state."""
    if not hasattr(env, "host_standup_phase"):
        env.host_standup_phase = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_reached"):
        env.host_standup_reached = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "host_standup_candidate_steps"):
        env.host_standup_candidate_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_transition_steps"):
        env.host_standup_transition_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_post_task_steps"):
        env.host_post_task_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    asset: Entity = env.scene[asset_cfg.name]
    head_idx = list(asset.site_names).index("head_link")
    head_height = asset.data.site_pos_w[env_ids, head_idx, 2]
    standing = (
        head_height > min_head_height
    )
    if not no_orientation:
        standing &= asset.data.projected_gravity_b[env_ids, 2] < upright_gravity_z

    phase = env.host_standup_phase[env_ids]
    candidate_steps_now = torch.where(
        (phase == STANDUP_PHASE) & standing,
        env.host_standup_candidate_steps[env_ids] + 1,
        torch.where(
            phase == STANDUP_PHASE,
            torch.zeros_like(env.host_standup_candidate_steps[env_ids]),
            env.host_standup_candidate_steps[env_ids],
        ),
    )
    entered_transition = (phase == STANDUP_PHASE) & (
        candidate_steps_now >= candidate_steps
    )
    next_phase = torch.where(
        entered_transition,
        torch.full_like(phase, TRANSITION_PHASE),
        phase,
    )
    # A failed transition or post-task phase immediately returns to STANDUP.
    next_phase = torch.where(
        (phase != STANDUP_PHASE) & ~standing,
        torch.full_like(next_phase, STANDUP_PHASE),
        next_phase,
    )
    transition_steps = torch.where(
        entered_transition,
        torch.zeros_like(env.host_standup_transition_steps[env_ids]),
        torch.where(
            (next_phase == TRANSITION_PHASE) & standing,
            env.host_standup_transition_steps[env_ids] + 1,
            torch.zeros_like(env.host_standup_transition_steps[env_ids]),
        ),
    )
    enter_post_task = (next_phase == TRANSITION_PHASE) & (
        transition_steps >= transition_steps_required
    )
    next_phase = torch.where(
        enter_post_task,
        torch.full_like(next_phase, POST_TASK_PHASE),
        next_phase,
    )
    env.host_standup_phase[env_ids] = next_phase
    env.host_standup_candidate_steps[env_ids] = torch.where(
        next_phase == STANDUP_PHASE,
        candidate_steps_now,
        torch.zeros_like(candidate_steps_now),
    )
    env.host_standup_transition_steps[env_ids] = transition_steps
    env.host_post_task_steps[env_ids] = torch.where(
        next_phase == POST_TASK_PHASE,
        env.host_post_task_steps[env_ids] + 1,
        torch.zeros_like(env.host_post_task_steps[env_ids]),
    )
    env.host_standup_reached[env_ids] |= entered_transition
