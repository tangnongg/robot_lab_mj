from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import UniformRpyBaseHeightCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_base_orientation_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(UniformRpyBaseHeightCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.desired_quat_w, asset.data.root_link_quat_w)
  return torch.exp(-(error**2) / (std**2))


def track_base_height_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(UniformRpyBaseHeightCommand, env.command_manager.get_term(command_name))
  error = asset.data.root_link_pos_w[:, 2] - command.command[:, 3]
  return torch.exp(-(error**2) / (std**2))


def base_xy_position_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  delta_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  return torch.sum(torch.square(delta_xy), dim=1)


def base_lin_vel_xy_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)


def base_lin_vel_z_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_ang_vel_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


def feet_contact_count_exp(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  expected_contacts: int = 4,
  contact_threshold: float = 1.0,
  std: float = 1.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data

  if data.force_history is not None:
    contact = (torch.norm(data.force_history, dim=-1) > contact_threshold).any(dim=-1)
  else:
    assert data.found is not None
    contact = data.found > 0

  contact_error = contact.float().sum(dim=1) - float(expected_contacts)
  return torch.exp(-(contact_error**2) / (std**2))


def undesired_contacts(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data

  if data.force_history is not None:
    contact = (torch.norm(data.force_history, dim=-1) > threshold).any(dim=-1)
  else:
    assert data.found is not None
    contact = data.found > 0

  return contact.float().sum(dim=1)

def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1)
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost
