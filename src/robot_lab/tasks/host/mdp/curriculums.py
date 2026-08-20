"""Adaptive single-course curriculum for the HoST stand-up task."""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _resolve_reset_ids(
    env: "ManagerBasedRlEnv", env_ids: Sequence[int] | slice
) -> torch.Tensor:
    """Convert the curriculum manager's reset selector to index tensor."""
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _level_values(
    level: torch.Tensor,
    initial_force: float,
    initial_action_rescale: float,
    min_action_rescale: float,
    initial_hold_steps: int,
    final_hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map the shared curriculum level linearly to all task parameters."""
    level = level.clamp(0.0, 1.0)
    force = initial_force * (1.0 - level)
    action_rescale = initial_action_rescale - (
        initial_action_rescale - min_action_rescale
    ) * level
    hold_steps = torch.round(
        initial_hold_steps + (final_hold_steps - initial_hold_steps) * level
    ).to(dtype=torch.long)
    return force, action_rescale, hold_steps


def _initialize_state(
    env: "ManagerBasedRlEnv",
    initial_force: float,
    initial_action_rescale: float,
    initial_hold_steps: int,
    initial_hold_settle_steps: int,
) -> None:
    """Create persistent global course state and per-environment parameters."""
    if not hasattr(env, "host_curriculum_level"):
        initial_level = float(os.environ.get("HOST_INITIAL_CURRICULUM_LEVEL", "0.0"))
        env.host_curriculum_level = torch.tensor(
            initial_level, dtype=torch.float32, device=env.device
        ).clamp_(0.0, 1.0)
        env.host_curriculum_window_successes = torch.zeros(
            (), dtype=torch.long, device=env.device
        )
        env.host_curriculum_window_episodes = torch.zeros(
            (), dtype=torch.long, device=env.device
        )
        env.host_curriculum_last_success_rate = torch.zeros((), device=env.device)
        env.host_curriculum_last_change = torch.zeros((), device=env.device)
        env.host_curriculum_completed_windows = torch.zeros(
            (), dtype=torch.long, device=env.device
        )

    if not hasattr(env, "host_traction_force"):
        env.host_traction_force = torch.full(
            (env.num_envs,), initial_force, dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "host_action_rescale"):
        env.host_action_rescale = torch.full(
            (env.num_envs,),
            initial_action_rescale,
            dtype=torch.float32,
            device=env.device,
        )
    if not hasattr(env, "host_standup_hold_required_steps"):
        env.host_standup_hold_required_steps = torch.full(
            (env.num_envs,), initial_hold_steps, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_standup_hold_settle_steps"):
        env.host_standup_hold_settle_steps = torch.full(
            (env.num_envs,),
            initial_hold_settle_steps,
            dtype=torch.long,
            device=env.device,
        )


def traction_force_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: Sequence[int] | slice,
    initial_force: float = 200.0,
    initial_action_rescale: float = 1.0,
    min_action_rescale: float = 0.25,
    initial_hold_steps: int = 5,
    final_hold_steps: int = 250,
    initial_hold_settle_steps: int = 0,
    final_hold_settle_steps: int = 35,
    window_episodes: int = 8192,
    promote_success_rate: float = 0.65,
    demote_success_rate: float = 0.35,
    promote_level_step: float = 0.025,
    demote_level_step: float = 0.0125,
    quiet_root_ang_vel_initial: float = 3.0,
    quiet_root_ang_vel_final: float = 1.0,
    quiet_root_lin_vel_initial: float = 2.0,
    quiet_root_lin_vel_final: float = 0.6,
    quiet_joint_vel_initial: float = 8.0,
    quiet_joint_vel_final: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Adapt one shared linear course from completed continuous holds.

    The task has only one course variable, ``level``.  It determines traction
    force, action rescale, and continuous-hold target together through linear
    mappings.  A level is held fixed for a full episode window.  It advances
    only after robust completion of that window and retreats after persistent
    failure, so post-task stabilization has time to adapt before assistance is
    removed further.
    """
    _initialize_state(
        env,
        initial_force,
        initial_action_rescale,
        initial_hold_steps,
        initial_hold_settle_steps,
    )
    reset_ids = _resolve_reset_ids(env, env_ids)

    # A fixed level is useful for the second phase of training: once a
    # reliable end-to-end actor exists, changing difficulty while the critic
    # is being rebuilt creates an avoidable distribution shift.  It is
    # opt-in, so normal curriculum training remains adaptive.
    fixed_level = os.environ.get("HOST_FIXED_CURRICULUM_LEVEL")
    if fixed_level is not None:
        env.host_curriculum_level.fill_(float(fixed_level))
        env.host_curriculum_window_successes.zero_()
        env.host_curriculum_window_episodes.zero_()

    hold_reached = getattr(env, "host_standup_hold_reached", None)
    if fixed_level is None and hold_reached is not None and reset_ids.numel() > 0:
        env.host_curriculum_window_successes += hold_reached[reset_ids].sum()
        env.host_curriculum_window_episodes += reset_ids.numel()

        if env.host_curriculum_window_episodes >= window_episodes:
            success_rate = (
                env.host_curriculum_window_successes.float()
                / env.host_curriculum_window_episodes.float()
            )
            env.host_curriculum_last_success_rate.copy_(success_rate)
            previous_level = env.host_curriculum_level.clone()
            if success_rate >= promote_success_rate:
                env.host_curriculum_level.add_(promote_level_step).clamp_(0.0, 1.0)
            elif success_rate < demote_success_rate:
                env.host_curriculum_level.sub_(demote_level_step).clamp_(0.0, 1.0)

            env.host_curriculum_last_change.copy_(
                env.host_curriculum_level - previous_level
            )
            if (
                env.host_curriculum_level >= 1.0
                and success_rate >= promote_success_rate
            ):
                env.host_curriculum_completed_windows.add_(1)
            else:
                env.host_curriculum_completed_windows.zero_()
            env.host_curriculum_window_successes.zero_()
            env.host_curriculum_window_episodes.zero_()

    force, action_rescale, hold_steps = _level_values(
        env.host_curriculum_level,
        initial_force,
        initial_action_rescale,
        min_action_rescale,
        initial_hold_steps,
        final_hold_steps,
    )
    level = env.host_curriculum_level.clamp(0.0, 1.0)
    hold_settle_steps = torch.round(
        initial_hold_settle_steps
        + (final_hold_settle_steps - initial_hold_settle_steps) * level
    ).to(dtype=torch.long)
    quiet_root_ang_vel = quiet_root_ang_vel_initial + (
        quiet_root_ang_vel_final - quiet_root_ang_vel_initial
    ) * level
    quiet_root_lin_vel = quiet_root_lin_vel_initial + (
        quiet_root_lin_vel_final - quiet_root_lin_vel_initial
    ) * level
    quiet_joint_vel = quiet_joint_vel_initial + (
        quiet_joint_vel_final - quiet_joint_vel_initial
    ) * level
    # Apply the current shared difficulty only to fresh episodes.  Existing
    # rollouts finish under the parameters they started with.
    if reset_ids.numel() > 0:
        env.host_traction_force[reset_ids] = force
        env.host_action_rescale[reset_ids] = action_rescale
        env.host_standup_hold_required_steps[reset_ids] = hold_steps
        env.host_standup_hold_settle_steps[reset_ids] = hold_settle_steps
        # Optional warm-start overrides keep a previously learned actor at
        # its deployment control scale while the new hold shaping is tuned.
        override_force = os.environ.get("HOST_OVERRIDE_FORCE")
        override_action = os.environ.get("HOST_OVERRIDE_ACTION_RESCALE")
        if override_force is not None:
            env.host_traction_force[reset_ids] = float(override_force)
        if override_action is not None:
            env.host_action_rescale[reset_ids] = float(override_action)

    return {
        "level": env.host_curriculum_level,
        "target_force": force,
        "target_action_rescale": action_rescale,
        "target_hold_required_steps": hold_steps.float(),
        "target_hold_settle_steps": hold_settle_steps.float(),
        "target_quiet_root_ang_vel": quiet_root_ang_vel,
        "target_quiet_root_lin_vel": quiet_root_lin_vel,
        "target_quiet_joint_vel": quiet_joint_vel,
        "mean_force": env.host_traction_force.mean(),
        "mean_action_rescale": env.host_action_rescale.mean(),
        "mean_hold_required_steps": env.host_standup_hold_required_steps.float().mean(),
        "mean_hold_settle_steps": env.host_standup_hold_settle_steps.float().mean(),
        "window_success_rate": env.host_curriculum_last_success_rate,
        "window_episodes": env.host_curriculum_window_episodes.float(),
        "last_level_change": env.host_curriculum_last_change,
        "completed_windows": env.host_curriculum_completed_windows.float(),
    }
