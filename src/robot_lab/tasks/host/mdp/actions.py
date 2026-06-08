"""Custom action term: Relative joint position control with unactuated gating.

Matches the official HoST implementation:
- *Relative* position control (target = current_joint_pos + action * scale)
- Actions zeroed during unactuated period (first N steps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import TransmissionType

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class RelativeJointPositionAction(JointPositionAction):
    """Joint position action using RELATIVE targets (current pos + action).

    In the official HoST implementation, the PD target is::

        target = current_dof_pos + action * action_rescale

    This matches that behaviour by setting the offset to the current joint
    position at each step (instead of ``default_joint_pos``).  During the
    unactuated period (episode_length <= unactuated_steps), actions are forced
    to zero so the robot is purely under traction-force control.
    """

    cfg: "RelativeJointPositionActionCfg"

    def __init__(
        self, cfg: "RelativeJointPositionActionCfg", env: "ManagerBasedRlEnv"
    ):
        super().__init__(cfg=cfg, env=env)

        self._unactuated_steps = cfg.unactuated_steps
        self._current_offset = torch.zeros_like(self._offset)

        # Ensure unactuated buffer exists on env
        if not hasattr(env, "host_unactuated_steps"):
            env.host_unactuated_steps = cfg.unactuated_steps

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply relative offset and unactuated gating before scale/offset."""
        # Gate: zero actions during unactuated period
        active = (
            self._env.episode_length_buf > self._unactuated_steps
        ).float().unsqueeze(-1)
        actions = actions * active

        # Relative control: offset = current joint position
        self._current_offset[:] = self._entity.data.joint_pos[
            :, self._target_ids
        ]
        self._offset = self._current_offset

        super().process_actions(actions)


@dataclass(kw_only=True)
class RelativeJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for relative joint position control.

    All parameters from ``JointPositionActionCfg`` are inherited.
    ``use_default_offset`` is forced to ``True`` internally (the *offset* is
    set to the *current* joint position at each step, not the default).
    """

    unactuated_steps: int = 30

    def __post_init__(self):
        # Must be joint-level control
        self.transmission_type = TransmissionType.JOINT
        # use_default_offset is forced True — we override _offset every step
        self.use_default_offset = True

    def build(self, env: "ManagerBasedRlEnv") -> RelativeJointPositionAction:
        return RelativeJointPositionAction(self, env)
