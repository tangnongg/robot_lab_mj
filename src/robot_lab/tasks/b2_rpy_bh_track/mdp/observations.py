from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

from .commands import UniformRpyBaseHeightCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_rpy(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  return torch.stack((roll, pitch, yaw), dim=-1)


def base_height(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def pose_command_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(UniformRpyBaseHeightCommand, env.command_manager.get_term(command_name))

  roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  rpy_error = torch.stack((roll, pitch, yaw), dim=-1) - command.command[:, :3]
  rpy_error[:, 2] = wrap_to_pi(rpy_error[:, 2])

  height_error = asset.data.root_link_pos_w[:, 2] - command.command[:, 3]
  return torch.cat((rpy_error, height_error.unsqueeze(-1)), dim=-1)
