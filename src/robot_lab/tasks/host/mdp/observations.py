"""Custom observation terms for HoST stand-up task.

All actor observations are zeroed during the unactuated period (first
``unactuated_steps`` environment steps) so the policy receives no feedback
while the robot is purely under traction-force control.

**Important:** Noise is applied *inside* these functions (before zero-gating),
matching the official HoST order: ``(raw + noise) * zero_mask``.
The ``noise`` parameter on ``ObservationTermCfg`` must be left as ``None``
for these terms, otherwise mjlab would add a second noise layer on top of the
already-zeroed result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Helper — zero output during unactuated period
# ---------------------------------------------------------------------------


def _zero_during_unactuated(
    env: "ManagerBasedRlEnv",
    result: torch.Tensor,
) -> torch.Tensor:
    """Zero *result* if the environment is still in the unactuated phase."""
    steps = getattr(env, "host_unactuated_steps", 30)
    # episode_length_buf is incremented BEFORE observation computation,
    # so episode_length_buf <= steps means still in unactuated phase.
    active = (env.episode_length_buf > steps).float()
    while active.ndim < result.ndim:
        active = active.unsqueeze(-1)
    return result * active


def _add_noise(
    result: torch.Tensor,
    n_min: float,
    n_max: float,
) -> torch.Tensor:
    """Add uniform noise in [n_min, n_max] to *result*."""
    return result + (torch.rand_like(result) * (n_max - n_min) + n_min)


# ---------------------------------------------------------------------------
# Actor observation terms (noise applied before unactuated gating)
# ---------------------------------------------------------------------------


def base_ang_vel(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_min: float = -0.05,
    noise_max: float = 0.05,
) -> torch.Tensor:
    """Base angular velocity, scaled by 0.25, with noise, zeroed during unactuated."""
    from mjlab.envs import mdp as envs_mdp
    raw = envs_mdp.base_ang_vel(env) * 0.25
    raw = _add_noise(raw, noise_min, noise_max)
    return _zero_during_unactuated(env, raw)


def projected_gravity(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_min: float = -0.05,
    noise_max: float = 0.05,
) -> torch.Tensor:
    """Projected gravity vector with noise, zeroed during unactuated."""
    asset: Entity = env.scene[asset_cfg.name]
    raw = asset.data.projected_gravity_b
    raw = _add_noise(raw, noise_min, noise_max)
    return _zero_during_unactuated(env, raw)


def joint_pos(
    env: "ManagerBasedRlEnv",
    biased: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*_joint",)),
    noise_min: float = -0.01,
    noise_max: float = 0.01,
) -> torch.Tensor:
    """Joint positions rel. to default, with noise, zeroed during unactuated."""
    from mjlab.envs import mdp as envs_mdp
    raw = envs_mdp.joint_pos_rel(env, biased=biased, asset_cfg=asset_cfg)
    raw = _add_noise(raw, noise_min, noise_max)
    return _zero_during_unactuated(env, raw)


def joint_vel(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*_joint",)),
    noise_min: float = -0.075,
    noise_max: float = 0.075,
) -> torch.Tensor:
    """Joint velocities rel. to default, scaled by 0.05, with noise, zeroed during unactuated."""
    from mjlab.envs import mdp as envs_mdp
    raw = envs_mdp.joint_vel_rel(env, asset_cfg=asset_cfg) * 0.05
    raw = _add_noise(raw, noise_min, noise_max)
    return _zero_during_unactuated(env, raw)


def last_action(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    """Last action (zeroed during unactuated period, no noise)."""
    result = env.action_manager.prev_action.clone()
    return _zero_during_unactuated(env, result)


def action_rescale_obs(
    env: "ManagerBasedRlEnv",
    initial_rescale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current action rescale factor with noise ±0.025, zeroed during unactuated."""
    if not hasattr(env, "host_action_rescale"):
        env.host_action_rescale = torch.full(
            (env.num_envs,), initial_rescale, device=env.device
        )
    noise = (torch.rand(env.num_envs, 1, device=env.device) - 0.5) * 0.05
    result = env.host_action_rescale.unsqueeze(1) + noise
    return _zero_during_unactuated(env, result)


def critic_phase(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Training-only phase bit used to select the two value heads."""
    if not hasattr(env, "host_standup_success"):
        env.host_standup_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    return env.host_standup_success.to(dtype=torch.float32).unsqueeze(-1)
