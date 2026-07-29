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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the per-episode stand-up success latch."""
    if not hasattr(env, "host_standup_success"):
        env.host_standup_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env.host_standup_success[env_ids] = False


def update_standup_success(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    threshold_height: float = 0.65,
    upright_gravity_z: float = -0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Latch whether each environment has reached the standing stage."""
    if not hasattr(env, "host_standup_success"):
        env.host_standup_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    asset: Entity = env.scene[asset_cfg.name]
    standing = (
        asset.data.root_link_pos_w[env_ids, 2] > threshold_height
    ) & (asset.data.projected_gravity_b[env_ids, 2] < upright_gravity_z)
    env.host_standup_success[env_ids] |= standing
