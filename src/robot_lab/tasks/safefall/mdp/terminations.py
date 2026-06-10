"""Termination conditions for SafeFall task.

Migrated from IsaacLab to mjlab.  API changes:
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


def fall_completed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    height_threshold: float = 0.15,
    velocity_threshold: float = 0.3,
) -> torch.Tensor:
    """Terminate when the robot has landed (low height + low velocity).

    Paper uses fixed episode length from fall detection to ground impact.
    We terminate when the robot is near ground and mostly stationary.
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    base_vel = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
    low_height = base_height < height_threshold
    low_vel = base_vel < velocity_threshold
    return low_height & low_vel


def joint_velocity_exceeded(
    env: ManagerBasedRlEnv,
    threshold: float = 200.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate on extreme joint velocities (simulation instability)."""
    asset: Entity = env.scene[asset_cfg.name]
    max_vel = torch.max(torch.abs(asset.data.joint_vel), dim=-1)[0]
    return max_vel > threshold
