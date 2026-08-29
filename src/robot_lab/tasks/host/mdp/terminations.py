"""Custom termination conditions for HoST stand-up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_velocity_exceeded(
    env: "ManagerBasedRlEnv",
    threshold: float = 300.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if any joint velocity exceeds *threshold* (rad/s)."""
    asset: Entity = env.scene[asset_cfg.name]
    max_vel = torch.abs(asset.data.joint_vel).max(dim=1)[0]
    return max_vel > threshold


def base_velocity_exceeded(
    env: "ManagerBasedRlEnv",
    threshold: float = 20.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if base linear velocity norm exceeds *threshold* (m/s)."""
    asset: Entity = env.scene[asset_cfg.name]
    vel_norm = torch.norm(asset.data.root_link_lin_vel_w[:, :3], dim=-1)
    return vel_norm > threshold


def standup_lost(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.58,
    upright_gravity_z: float = -0.65,
    min_head_height: float = 1.00,
    lost_steps: int = 8,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """End an episode when a confirmed stand collapses back to kneeling.

    A short grace window avoids terminating on one noisy physics frame, while
    the height/orientation hysteresis makes a confirmed stand fail clearly
    once it returns to the kneeling pivot that caused the old spin behavior.
    """
    phase = getattr(env, "host_standup_phase", None)
    if phase is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    confirmed = phase > 0
    asset: Entity = env.scene[asset_cfg.name]
    head_idx = list(asset.site_names).index("head_link")
    head_height = asset.data.site_pos_w[:, head_idx, 2]
    standing = (head_height > min_head_height) & (
        asset.data.projected_gravity_b[:, 2] < upright_gravity_z
    )
    if not hasattr(env, "host_standup_lost_steps"):
        env.host_standup_lost_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    lost = confirmed & ~standing
    env.host_standup_lost_steps = torch.where(
        lost,
        env.host_standup_lost_steps + 1,
        torch.zeros_like(env.host_standup_lost_steps),
    )
    return confirmed & (env.host_standup_lost_steps >= lost_steps)
