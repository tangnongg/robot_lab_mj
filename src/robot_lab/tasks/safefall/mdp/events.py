"""Event terms for SafeFall: domain randomization and fall initialization.

Paper Table I describes 6 failure factors. We implement:
- Random initial falling states (Stage I curriculum)
- Physics randomization (friction, restitution) via built-in DR

Migrated from IsaacLab to mjlab.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_falling_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    height_range: tuple[float, float] = (1.0, 1.2),
    horizontal_speed_range: tuple[float, float] = (0.0, 2.0),
    downward_speed_range: tuple[float, float] = (-3.0, -1.5),
    ang_vel_range: tuple[float, float] = (-3.0, 3.0),
):
    """Reset robot to random falling configurations.

    Paper Stage I: place robot in random poses deviating from default,
    at various heights above ground with random orientations and downward velocity.

    Args:
        env: The manager-based RL environment.
        env_ids: Environment indices to reset.
        asset_cfg: Scene entity configuration for the robot.
        height_range: (min, max) range for base height above ground (m).
        horizontal_speed_range: Horizontal speed magnitude range (m/s).
        downward_speed_range: Downward root velocity range (m/s).
        ang_vel_range: (min, max) range for angular velocity components (rad/s).
    """
    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # Start from default root state.
    root_state = asset.data.default_root_state[env_ids].clone()  # (N, 13)

    # Random height.
    heights = torch.empty(num_envs, device=device).uniform_(*height_range)
    root_state[:, 2] = heights

    # Uniform random orientations cover forward, backward, and lateral falls.
    quat = torch.randn(num_envs, 4, device=device)
    root_state[:, 3:7] = quat / torch.linalg.vector_norm(
        quat, dim=-1, keepdim=True
    ).clamp(min=1.0e-6)

    # Random horizontal direction and explicitly downward velocity. Starting
    # above the robot's collision radius avoids reset penetration, while this
    # speed ensures impact occurs well inside the fixed 0.8 s episode.
    heading = torch.empty(num_envs, device=device).uniform_(-torch.pi, torch.pi)
    speed = torch.empty(num_envs, device=device).uniform_(*horizontal_speed_range)
    root_state[:, 7] = speed * torch.cos(heading)
    root_state[:, 8] = speed * torch.sin(heading)
    root_state[:, 9] = torch.empty(num_envs, device=device).uniform_(
        *downward_speed_range
    )

    # Random angular velocity.
    root_state[:, 10] = torch.empty(num_envs, device=device).uniform_(*ang_vel_range)
    root_state[:, 11] = torch.empty(num_envs, device=device).uniform_(*ang_vel_range)
    root_state[:, 12] = torch.empty(num_envs, device=device).uniform_(*ang_vel_range)

    # Apply env_origins offset to positions.
    root_state[:, 0:3] += env.scene.env_origins[env_ids]

    asset.write_root_state_to_sim(root_state, env_ids)

    # Randomize joint positions slightly.
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(-0.2, 0.2)
    limits = asset.data.joint_pos_limits[env_ids]
    joint_pos = torch.minimum(
        torch.maximum(joint_pos, limits[..., 0] + 0.02),
        limits[..., 1] - 0.02,
    )
    joint_vel = torch.empty_like(joint_pos).uniform_(-0.5, 0.5)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Stage II — sample from predictor-flagged state bank
# ---------------------------------------------------------------------------

_STAGE2_BANK: dict[str, torch.Tensor] | None = None
_STAGE2_BANK_PATH: Path | None = None


def _load_stage2_bank(bank_path: str) -> dict[str, torch.Tensor]:
    """Lazy‑load the Stage II state bank (shared across all envs)."""
    global _STAGE2_BANK, _STAGE2_BANK_PATH
    resolved = Path(bank_path).resolve()
    if _STAGE2_BANK is None or _STAGE2_BANK_PATH != resolved:
        _STAGE2_BANK = torch.load(resolved, map_location="cpu", weights_only=False)
        _STAGE2_BANK_PATH = resolved
    if _STAGE2_BANK.get("state_schema_version", 0) < 2:
        raise ValueError(
            f"Stage II bank {resolved} does not contain exact root states. "
            "Rebuild it with prepare_stage2_states.py."
        )
    return _STAGE2_BANK


def reset_falling_from_bank(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    bank_path: str = "models/stage2_state_bank.pt",
    pos_noise: float = 0.05,
    vel_noise: float = 0.1,
):
    """Reset robot to predictor-flagged falling states (paper Stage II).

    Samples exact initial states from a pre-extracted bank of realistic
    falling configurations, then applies small joint noise for diversity.

    Args:
        env: The manager-based RL environment.
        env_ids: Environment indices to reset.
        asset_cfg: Scene entity configuration for the robot.
        bank_path: Path to the ``.pt`` state bank file.
        pos_noise: Std of Gaussian noise added to joint positions (rad).
        vel_noise: Std of Gaussian noise added to joint velocities (rad/s).
    """
    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    bank = _load_stage2_bank(bank_path)

    K: int = int(bank["K"])
    idx = torch.randint(0, K, (num_envs,))

    root_state = bank["root_state"][idx].to(device).clone()  # (N, 13)
    joint_pos = bank["joint_pos"][idx].to(device).clone()    # (N, 29)
    joint_vel = bank["joint_vel"][idx].to(device).clone()    # (N, 29)

    # XY translation is dynamically irrelevant on a plane. Preserve the exact
    # height, orientation, and velocities captured when the predictor fired.
    root_state[:, 0:2] = 0.0

    # Small Gaussian noise on joints for diversity.
    joint_pos += torch.randn_like(joint_pos) * pos_noise
    joint_vel += torch.randn_like(joint_vel) * vel_noise
    limits = asset.data.joint_pos_limits[env_ids]
    joint_pos = torch.minimum(
        torch.maximum(joint_pos, limits[..., 0] + 0.02),
        limits[..., 1] - 0.02,
    )

    root_state[:, 0:3] += env.scene.env_origins[env_ids]
    asset.write_root_state_to_sim(root_state, env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
