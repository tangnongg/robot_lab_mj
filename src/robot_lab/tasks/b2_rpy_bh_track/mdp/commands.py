from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_error_magnitude,
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformRpyBaseHeightCommand(CommandTerm):
  """Uniformly sampled absolute base pose command."""

  cfg: "UniformRpyBaseHeightCommandCfg"

  def __init__(self, cfg: "UniformRpyBaseHeightCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    self.env = env
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
    base_height_offset = sample_uniform(*ranges.base_height_offset, (n,), device=self.device)

    self._command[env_ids, 0] = roll
    self._command[env_ids, 1] = pitch
    self._command[env_ids, 2] = yaw
    self._command[env_ids, 3] = base_height_offset + self.robot.data.default_root_state[env_ids, 2]
    self._desired_quat_w[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)

  def _update_command(self) -> None:
    return

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    env_origins = self._env.scene.env_origins.cpu().numpy() # TODO：check if real still/init state of base_link is same as ideal state
    current_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    current_pos_ws[:, 1] += 0.3  # offset the current position visualization for better visibility
    current_rot_ws = matrix_from_quat(self.robot.data.root_link_quat_w).cpu().numpy()
    target_rot_ws = matrix_from_quat(self._desired_quat_w).cpu().numpy()
    cmds = self._command.cpu().numpy()
    frame_scale = self.cfg.viz.frame_scale
    axis_radius = self.cfg.viz.axis_radius * visualizer.meansize
    error_arrow_width = self.cfg.viz.error_arrow_width * visualizer.meansize

    for batch in env_indices:
      target_pos_w = env_origins[batch].copy()
      target_pos_w[1] += 0.3
      target_pos_w[2] = cmds[batch, 3]

      visualizer.add_frame(
        position=target_pos_w,
        rotation_matrix=target_rot_ws[batch],
        scale=frame_scale,
        axis_radius=axis_radius,
        alpha=self.cfg.viz.frame_alpha,
        axis_colors=self.cfg.viz.frame_axis_colors,
        label=f"target_pose_frame_{batch}",
      )

      current_pos_w = current_pos_ws[batch].copy()
      visualizer.add_frame(
        position=current_pos_w,
        rotation_matrix=current_rot_ws[batch],
        scale=frame_scale,
        axis_radius=axis_radius,
        alpha=float(self.cfg.viz.frame_alpha / 2.0), # TODO: does not work
        axis_colors=tuple(self.cfg.viz.frame_axis_colors),
        label=f"current_pose_frame_{batch}",
      )

      if np.linalg.norm(current_pos_w - target_pos_w) > 1.0e-6:
        visualizer.add_arrow(
          start=current_pos_w,
          end=target_pos_w,
          color=self.cfg.viz.error_arrow_color,
          width=error_arrow_width,
          label=f"pose_error_{batch}",
        )


@dataclass(kw_only=True)
class UniformRpyBaseHeightCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    roll: tuple[float, float] = (-0.80, 0.80)
    pitch: tuple[float, float] = (-0.30, 0.30)
    yaw: tuple[float, float] = (-0.40, 0.40)
    base_height_offset: tuple[float, float] = (0.2, 0.05)

  @dataclass
  class VizCfg:
    frame_scale: float = 0.3
    axis_radius: float = 0.2
    frame_alpha: float = 1.0
    error_arrow_width: float = 0.2
    error_arrow_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.65)
    frame_axis_colors: tuple[tuple[float, float, float], ...] = (
      (1.0, 0.0, 0.0),
      (0.0, 1.0, 0.0),
      (0.0, 0.0, 1.0),
    )

  entity_name: str
  ranges: Ranges = field(default_factory=Ranges)
  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> UniformRpyBaseHeightCommand:
    return UniformRpyBaseHeightCommand(self, env)
