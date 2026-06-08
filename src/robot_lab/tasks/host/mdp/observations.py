"""Custom observation terms for HoST stand-up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def projected_gravity(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Projected gravity vector in the robot's base frame (x,y,z)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def action_rescale_obs(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation of current action rescale factor (with uniform noise ±0.025)."""
    if not hasattr(env, "host_action_rescale"):
        env.host_action_rescale = torch.ones(env.num_envs, device=env.device)
    noise = (torch.rand(env.num_envs, 1, device=env.device) - 0.5) * 0.05
    return env.host_action_rescale.unsqueeze(1) + noise
