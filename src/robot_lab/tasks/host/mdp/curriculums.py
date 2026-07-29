"""Traction force curriculum for HoST stand-up task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def traction_force_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: Sequence[int],
    initial_force: float = 100.0,
    force_decrement: float = 20.0,
    action_rescale_decrement: float = 0.02,
    initial_action_rescale: float = 1.0,
    min_action_rescale: float = 0.25,
    threshold_height: float = 0.65,
    no_orientation: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Update the curriculum when the robot ends an episode standing.

    The G1 MJCF has no separate head body, so using ``torso_link`` with the
    paper's head-height threshold makes success unreachable. We use the paper's
    stage-3 base height together with upright projected gravity instead. On
    success, the upward traction force is reduced by
    *force_decrement* (clamped to >= 0) and the action rescale factor is
    reduced by *action_rescale_decrement* (clamped to >= *min_action_rescale*).

    Args:
        env: The manager-based RL environment.
        env_ids: Environment indices being reset (curriculum trigger).
        initial_force: Initial traction force (N) for new environments.
        force_decrement: How much to reduce force on success (N).
        action_rescale_decrement: How much to reduce action scale on success.
        initial_action_rescale: Initial action scale for new environments.
        min_action_rescale: Minimum action rescale factor.
        threshold_height: Base height (m) required at the end of the episode.
        no_orientation: If True, skip orientation check.
        asset_cfg: Scene entity configuration for the robot.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Initialize each curriculum buffer independently. Observation construction
    # may create the action-scale buffer before the force buffer exists.
    if not hasattr(env, "host_traction_force"):
        env.host_traction_force = torch.full(
            (env.num_envs,), initial_force, device=env.device
        )
    if not hasattr(env, "host_action_rescale"):
        env.host_action_rescale = torch.full(
            (env.num_envs,), initial_action_rescale, device=env.device
        )

    if len(env_ids) == 0:
        return {
            "mean_force": env.host_traction_force.mean(),
            "mean_action_rescale": env.host_action_rescale.mean(),
            "success_rate": torch.zeros((), device=env.device),
        }

    reset_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if hasattr(env, "host_standup_success"):
        success_mask = env.host_standup_success[reset_ids]
    else:
        base_height = asset.data.root_link_pos_w[reset_ids, 2]
        success_mask = base_height > threshold_height
        if not no_orientation:
            gravity_z = asset.data.projected_gravity_b[reset_ids, 2]
            success_mask &= gravity_z < -0.8
    if success_mask.any():
        success_ids = reset_ids[success_mask]
        env.host_traction_force[success_ids] = (
            env.host_traction_force[success_ids] - force_decrement
        ).clamp(min=0.0)
        env.host_action_rescale[success_ids] = (
            env.host_action_rescale[success_ids] - action_rescale_decrement
        ).clamp(min=min_action_rescale)

    return {
        "mean_force": env.host_traction_force.mean(),
        "mean_action_rescale": env.host_action_rescale.mean(),
        "success_rate": success_mask.float().mean(),
    }
