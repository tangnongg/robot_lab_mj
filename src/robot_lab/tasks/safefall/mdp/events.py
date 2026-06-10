"""Event terms for SafeFall: domain randomization and fall initialization.

Paper Table I describes 6 failure factors. We implement:
- Random initial falling states (Stage I curriculum)
- Physics randomization (friction, restitution) via built-in DR

Migrated from IsaacLab to mjlab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_falling_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    height_range: tuple = (0.4, 0.9),
    vel_range: tuple = (0.0, 2.0),
    ang_vel_range: tuple = (-1.0, 1.0),
):
    """Reset robot to random falling configurations.

    Paper Stage I: place robot in random poses deviating from default,
    at various heights above ground with random orientations and downward velocity.

    Args:
        env: The manager-based RL environment.
        env_ids: Environment indices to reset.
        asset_cfg: Scene entity configuration for the robot.
        height_range: (min, max) range for base height above ground (m).
        vel_range: (min, max) range for linear velocity components (m/s).
        ang_vel_range: (min, max) range for angular velocity components (rad/s).
    """
    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # Start from default root state.
    root_state = asset.data.default_root_state[env_ids].clone()  # (N, 13)

    # Random height.
    heights = torch.empty(num_envs, device=device).uniform_(*height_range)
    root_state[:, 2] = heights

    # Random tilt (roll, pitch) and yaw → quaternion (wxyz).
    roll = torch.empty(num_envs, device=device).uniform_(-0.8, 0.8)
    pitch = torch.empty(num_envs, device=device).uniform_(-0.8, 0.8)
    yaw = torch.empty(num_envs, device=device).uniform_(-3.14, 3.14)
    quat = quat_from_euler_xyz(roll, pitch, yaw)  # (N, 4) wxyz
    root_state[:, 3:7] = quat

    # Random downward velocity.
    root_state[:, 7] = torch.empty(num_envs, device=device).uniform_(-vel_range[1], vel_range[1])
    root_state[:, 8] = torch.empty(num_envs, device=device).uniform_(-vel_range[1], vel_range[1])
    root_state[:, 9] = torch.empty(num_envs, device=device).uniform_(-vel_range[1], -0.5)

    # Random angular velocity.
    root_state[:, 10] = torch.empty(num_envs, device=device).uniform_(*ang_vel_range)
    root_state[:, 11] = torch.empty(num_envs, device=device).uniform_(*ang_vel_range)
    root_state[:, 12] = torch.empty(num_envs, device=device).uniform_(*ang_vel_range)

    # Apply env_origins offset to positions.
    root_state[:, 0:3] += env.scene.env_origins[env_ids]

    asset.write_root_state_to_sim(root_state, env_ids)

    # Randomize joint positions slightly.
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(-0.2, 0.2)
    joint_vel = torch.empty_like(joint_pos).uniform_(-0.5, 0.5)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
