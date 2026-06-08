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
    min_action_rescale: float = 0.25,
    threshold_height: float = 0.9,
    head_body_name: str = "torso_link",
    foot_body_names: list[str] | None = None,
    no_orientation: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Update traction force curriculum based on head height achieved.

    When an environment achieves a head height (relative to feet) above
    *threshold_height*, the upward traction force is reduced by
    *force_decrement* (clamped to >= 0) and the action rescale factor is
    reduced by *action_rescale_decrement* (clamped to >= *min_action_rescale*).

    Args:
        env: The manager-based RL environment.
        env_ids: Environment indices being reset (curriculum trigger).
        initial_force: Initial traction force (N) for new environments.
        force_decrement: How much to reduce force on success (N).
        action_rescale_decrement: How much to reduce action scale on success.
        min_action_rescale: Minimum action rescale factor.
        threshold_height: Relative head height (m) above feet to trigger reduction.
        head_body_name: Name of the head body in the MJCF.
        foot_body_names: Names of foot bodies used for reference height.
        no_orientation: If True, skip orientation check.
        asset_cfg: Scene entity configuration for the robot.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Initialize buffers on first call
    if not hasattr(env, "host_traction_force"):
        env.host_traction_force = torch.full(
            (env.num_envs,), initial_force, device=env.device
        )
        env.host_action_rescale = torch.ones(env.num_envs, device=env.device)

    if len(env_ids) == 0:
        return

    reset_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # Compute head height relative to feet for resetting envs
    body_names = list(asset.body_names)
    head_idx = body_names.index(head_body_name)

    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_indices = [body_names.index(n) for n in foot_body_names]

    head_height = asset.data.body_link_pos_w[reset_ids, head_idx, 2]
    feet_height = (
        asset.data.body_link_pos_w[reset_ids][:, foot_indices, 2].mean(dim=-1)
    )
    relative_height = head_height - feet_height

    # Update curriculum for environments that exceeded the threshold
    success_mask = relative_height > threshold_height
    if success_mask.any():
        success_ids = reset_ids[success_mask]
        env.host_traction_force[success_ids] = (
            env.host_traction_force[success_ids] - force_decrement
        ).clamp(min=0.0)
        env.host_action_rescale[success_ids] = (
            env.host_action_rescale[success_ids] - action_rescale_decrement
        ).clamp(min=min_action_rescale)
