from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, UniformVelocityCommand

class UniformThresholdVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: torch.Tensor):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@dataclass(kw_only=True)
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


# TODO: add a fixed velocity command generator for play
# # Customized command generator for play, fixed velocity command 
# class FixedVelocityCommand(CommandTerm):
#     """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

#     cfg: FixedVelocityCommandCfg
#     """The configuration of the command generator."""

#     def _resample_command(self, env_ids: torch.Tensor):
#         # set small commands to zero
#         self.vel_command_b[env_ids, :2] *= torch.tensor([0.0, 0.0], dtype=torch.float32).to(self.vel_command_b.device


# @dataclass(kw_only=True)
# class FixedVelocityCommandCfg(CommandTerm):
#     """Configuration for the uniform threshold velocity command generator."""

#     class_type: type = FixedVelocityCommand

