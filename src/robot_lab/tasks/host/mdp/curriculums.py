"""Force/action curriculum for the HoST stand-up task."""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


DEFAULT_WINDOW_EPISODES = 8192
DEFAULT_PROMOTE_SUCCESS_RATE = 0.75


def _resolve_reset_ids(
    env: "ManagerBasedRlEnv", env_ids: Sequence[int] | slice
) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _initialize_state(
    env: "ManagerBasedRlEnv",
    initial_force: float,
    initial_action_rescale: float,
    window_episodes: int,
) -> None:
    """Create only the shared force/action course state."""
    if not hasattr(env, "host_curriculum_level"):
        initial_level = float(os.environ.get("HOST_INITIAL_CURRICULUM_LEVEL", "0.0"))
        env.host_curriculum_level = torch.tensor(
            initial_level, dtype=torch.float32, device=env.device
        ).clamp_(0.0, 1.0)
        env.host_curriculum_last_success_rate = torch.zeros((), device=env.device)
        env.host_curriculum_last_change = torch.zeros((), device=env.device)
    if not hasattr(env, "host_curriculum_window_episodes"):
        env.host_curriculum_window_episodes = torch.zeros(
            (), dtype=torch.long, device=env.device
        )
        env.host_curriculum_window_successes = torch.zeros(
            (), dtype=torch.long, device=env.device
        )
        env.host_curriculum_completed_windows = torch.zeros(
            (), dtype=torch.long, device=env.device
        )
    if not hasattr(env, "host_traction_force"):
        env.host_traction_force = torch.full(
            (env.num_envs,), initial_force, dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "host_action_rescale"):
        env.host_action_rescale = torch.full(
            (env.num_envs,), initial_action_rescale,
            dtype=torch.float32, device=env.device,
        )


def traction_force_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: Sequence[int] | slice,
    initial_force: float = 200.0,
    initial_action_rescale: float = 1.0,
    min_action_rescale: float = 0.25,
    window_episodes: int = DEFAULT_WINDOW_EPISODES,
    promote_success_rate: float = DEFAULT_PROMOTE_SUCCESS_RATE,
    promote_level_step: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Increase difficulty after each non-overlapping episode batch.

    Exactly ``window_episodes`` completed episodes are accumulated, their
    stand-up success rate is evaluated, and the counters are then cleared for
    the next batch. The curriculum is monotonic.
    """
    _initialize_state(env, initial_force, initial_action_rescale, window_episodes)
    reset_ids = _resolve_reset_ids(env, env_ids)

    fixed_level = os.environ.get("HOST_FIXED_CURRICULUM_LEVEL")
    if fixed_level is not None:
        env.host_curriculum_level.fill_(float(fixed_level))

    env.host_curriculum_last_change.zero_()
    standup_reached = getattr(env, "host_standup_reached", None)
    if standup_reached is not None and reset_ids.numel() > 0 and fixed_level is None:
        outcomes = standup_reached[reset_ids].bool()
        capacity = int(window_episodes)
        offset = 0
        while offset < outcomes.numel():
            remaining = capacity - int(env.host_curriculum_window_episodes.item())
            count = min(remaining, outcomes.numel() - offset)
            batch = outcomes[offset : offset + count]
            env.host_curriculum_window_episodes.add_(count)
            env.host_curriculum_window_successes.add_(batch.to(torch.long).sum())
            offset += count

            if int(env.host_curriculum_window_episodes.item()) == capacity:
                rate = (
                    env.host_curriculum_window_successes.float() / float(capacity)
                )
                env.host_curriculum_last_success_rate.copy_(rate)
                previous_level = env.host_curriculum_level.clone()
                if rate >= promote_success_rate:
                    env.host_curriculum_level.add_(promote_level_step).clamp_(0.0, 1.0)
                env.host_curriculum_last_change.copy_(
                    env.host_curriculum_level - previous_level
                )
                env.host_curriculum_completed_windows.add_(1)
                # The reported rate is retained, while counters start a new
                # independent batch immediately after it is evaluated.
                env.host_curriculum_window_episodes.zero_()
                env.host_curriculum_window_successes.zero_()

    level = env.host_curriculum_level.clamp(0.0, 1.0)
    force = initial_force * (1.0 - level)
    action_rescale = initial_action_rescale - (
        initial_action_rescale - min_action_rescale
    ) * level
    if reset_ids.numel() > 0:
        env.host_traction_force[reset_ids] = force
        env.host_action_rescale[reset_ids] = action_rescale
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
        "window_success_rate": env.host_curriculum_last_success_rate,
        "window_episodes": env.host_curriculum_window_episodes.float(),
        "last_level_change": env.host_curriculum_last_change,
    }
