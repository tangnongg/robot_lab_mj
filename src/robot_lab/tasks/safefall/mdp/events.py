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
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_falling_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    height_range: tuple[float, float] = (0.78, 0.86),
    horizontal_speed_range: tuple[float, float] = (0.3, 1.8),
    downward_speed_range: tuple[float, float] = (-0.35, 0.05),
    orientation_tilt_range: tuple[float, float] = (-0.18, 0.18),
    fall_angular_speed_range: tuple[float, float] = (1.5, 4.5),
    yaw_angular_speed_range: tuple[float, float] = (-1.0, 1.0),
    joint_noise: float = 0.05,
    joint_velocity_range: tuple[float, float] = (-0.3, 0.3),
):
    """Reset robot to physically plausible walking-instability states.

    Stage I is intended to approximate a walking controller losing balance.  The
    robot therefore starts close to its normal foot-supported height, with the
    head above the torso and only a small roll/pitch perturbation.  A horizontal
    velocity and an angular velocity about a random horizontal axis create the
    fall.  This avoids training on upside-down or high-altitude free-fall states.

    Args:
        env: The manager-based RL environment.
        env_ids: Environment indices to reset.
        asset_cfg: Scene entity configuration for the robot.
        height_range: (min, max) range for pelvis height above ground (m).
        horizontal_speed_range: Horizontal speed magnitude range (m/s).
        downward_speed_range: Downward root velocity range (m/s).
        orientation_tilt_range: roll and pitch range around the upright pose.
        fall_angular_speed_range: magnitude of horizontal-axis tipping speed.
        yaw_angular_speed_range: residual yaw-rate range.
    """
    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # Start from default root state.
    root_state = asset.data.default_root_state[env_ids].clone()  # (N, 13)

    # Random height.
    heights = torch.empty(num_envs, device=device).uniform_(*height_range)
    root_state[:, 2] = heights

    # Preserve the upright walking envelope.  Yaw is unconstrained, while
    # roll/pitch stay small enough that the head remains above the torso.
    roll = torch.empty(num_envs, device=device).uniform_(*orientation_tilt_range)
    pitch = torch.empty(num_envs, device=device).uniform_(*orientation_tilt_range)
    yaw = torch.empty(num_envs, device=device).uniform_(-torch.pi, torch.pi)
    root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

    # Preserve a walking-scale horizontal velocity and only a small vertical
    # component.  The fall is caused by the tipping angular velocity below,
    # rather than by dropping the whole robot from above the terrain.
    heading = torch.empty(num_envs, device=device).uniform_(-torch.pi, torch.pi)
    speed = torch.empty(num_envs, device=device).uniform_(*horizontal_speed_range)
    root_state[:, 7] = speed * torch.cos(heading)
    root_state[:, 8] = speed * torch.sin(heading)
    root_state[:, 9] = torch.empty(num_envs, device=device).uniform_(
        *downward_speed_range
    )

    # Tip about a random horizontal axis, as a walking fall would begin.  Do
    # not inject an arbitrary 3-D spin that could make the initial state
    # inverted before the policy has any opportunity to react.
    fall_heading = torch.empty(num_envs, device=device).uniform_(-torch.pi, torch.pi)
    fall_speed = torch.empty(num_envs, device=device).uniform_(*fall_angular_speed_range)
    root_state[:, 10] = fall_speed * torch.cos(fall_heading)
    root_state[:, 11] = fall_speed * torch.sin(fall_heading)
    root_state[:, 12] = torch.empty(num_envs, device=device).uniform_(*yaw_angular_speed_range)

    # Apply env_origins offset to positions.
    root_state[:, 0:3] += env.scene.env_origins[env_ids]

    asset.write_root_state_to_sim(root_state, env_ids)

    # Randomize joint positions slightly.
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(-joint_noise, joint_noise)
    limits = asset.data.joint_pos_limits[env_ids]
    joint_pos = torch.minimum(
        torch.maximum(joint_pos, limits[..., 0] + 0.02),
        limits[..., 1] - 0.02,
    )
    joint_vel = torch.empty_like(joint_pos).uniform_(*joint_velocity_range)
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
    upright_up_z_min: float = 0.2,
    root_height_range: tuple[float, float] = (0.55, 1.0),
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
    bank_root = bank["root_state"]
    world_up = torch.zeros((K, 3), dtype=bank_root.dtype)
    world_up[:, 2] = 1.0
    body_up = quat_apply(bank_root[:, 3:7], world_up)
    valid = (
        (body_up[:, 2] >= upright_up_z_min)
        & (bank_root[:, 2] >= root_height_range[0])
        & (bank_root[:, 2] <= root_height_range[1])
    )
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        raise ValueError(
            "Stage II state bank has no upright, near-ground states; rebuild it "
            "from walking-policy trajectories (state_schema_version=2)."
        )
    idx = valid_indices[torch.randint(valid_indices.numel(), (num_envs,))]

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
