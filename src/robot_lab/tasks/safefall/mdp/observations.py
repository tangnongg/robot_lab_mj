"""Custom observation terms for SafeFall task.

Paper observation space: pelvis orientation (r, p), joint states (q, q_dot),
previous actions a_{t-1}, angular velocity omega, projected gravity g_b,
stacked over 5 timesteps for temporal context.

Migrated from IsaacLab to mjlab.  API changes:
- asset.data.root_quat_w → asset.data.root_link_quat_w
- asset.data.root_pos_w → asset.data.root_link_pos_w
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pelvis_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pelvis roll and pitch angles from quaternion.

    Returns:
        Tensor of shape (num_envs, 2): (roll, pitch) in radians.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    return torch.stack([roll, pitch], dim=-1)


def base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Base height above ground.

    Returns:
        Tensor of shape (num_envs, 1): height in meters.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]
