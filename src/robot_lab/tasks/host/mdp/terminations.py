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
