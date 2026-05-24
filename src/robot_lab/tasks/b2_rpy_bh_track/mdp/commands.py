from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_error_magnitude,
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class UniformRpyBaseHeightCommand(CommandTerm):
  """Uniformly sampled absolute base pose command."""

  cfg: "UniformRpyBaseHeightCommandCfg"

  def __init__(self, cfg: "UniformRpyBaseHeightCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self._command = torch.zeros(self.num_envs, 4, device=self.device)
    self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
    self._desired_quat_w[:, 0] = 1.0

    self.metrics["error_orientation"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def desired_quat_w(self) -> torch.Tensor:
    return self._desired_quat_w

  def _update_metrics(self) -> None:
    self.metrics["error_orientation"] = quat_error_magnitude(
      self._desired_quat_w,
      self.robot.data.root_link_quat_w,
    )
    self.metrics["error_height"] = torch.abs(
      self.robot.data.root_link_pos_w[:, 2] - self._command[:, 3]
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    n = len(env_ids)
    ranges = self.cfg.ranges

    roll = sample_uniform(*ranges.roll, (n,), device=self.device)
    pitch = sample_uniform(*ranges.pitch, (n,), device=self.device)
    yaw = sample_uniform(*ranges.yaw, (n,), device=self.device)
    base_height = sample_uniform(*ranges.base_height, (n,), device=self.device)

    self._command[env_ids, 0] = roll
    self._command[env_ids, 1] = pitch
    self._command[env_ids, 2] = yaw
    self._command[env_ids, 3] = base_height
    self._desired_quat_w[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)

  def _update_command(self) -> None:
    return


@dataclass(kw_only=True)
class UniformRpyBaseHeightCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    roll: tuple[float, float] = (-0.30, 0.30)
    pitch: tuple[float, float] = (-0.30, 0.30)
    yaw: tuple[float, float] = (-0.60, 0.60)
    base_height: tuple[float, float] = (0.52, 0.64)

  entity_name: str
  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: "ManagerBasedRlEnv") -> UniformRpyBaseHeightCommand:
    return UniformRpyBaseHeightCommand(self, env)
