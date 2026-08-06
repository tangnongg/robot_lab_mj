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


def simulation_state_invalid(
    env: ManagerBasedRlEnv,
    max_abs_qpos: float = 1.0e3,
    max_abs_qvel: float = 200.0,
) -> torch.Tensor:
    """Detect non-finite or non-physical simulator state.

    This is a numerical safety boundary, not a task-success termination. A state
    beyond these limits has already left the mechanically meaningful regime and
    otherwise produces unbounded squared impact costs before the next observation.
    """
    qpos = env.sim.data.qpos
    qvel = env.sim.data.qvel
    finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
    bounded = (qpos.abs().amax(dim=-1) <= max_abs_qpos) & (
        qvel.abs().amax(dim=-1) <= max_abs_qvel
    )
    return ~(finite & bounded)
