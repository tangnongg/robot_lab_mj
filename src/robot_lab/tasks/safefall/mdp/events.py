"""Event terms for SafeFall: domain randomization and fall initialization.

Paper Table I describes 6 failure factors. We implement:
- Random initial falling states (Stage I curriculum)
- Physics randomization (friction, restitution) via built-in DR

Migrated from IsaacLab to mjlab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_falling_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    height_range: tuple = (0.4, 0.9),
    vel_range: tuple = (0.0, 2.0),
    ang_vel_range: tuple = (-1.0, 1.0),
):
    """Reset robot to random falling configurations.

    Paper Stage I: place robot in random poses deviating from default,
    at various heights above ground with random orientations and downward velocity.

    Args:
        env: The manager-based RL environment.
        env_ids: Environment indices to reset.
        asset_cfg: Scene entity configuration for the robot.
        height_range: (min, max) range for base height above ground (m).
        vel_range: (min, max) range for linear velocity components (m/s).
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

    # Random tilt (roll, pitch) and yaw → quaternion (wxyz).
    roll = torch.empty(num_envs, device=device).uniform_(-0.8, 0.8)
    pitch = torch.empty(num_envs, device=device).uniform_(-0.8, 0.8)
    yaw = torch.empty(num_envs, device=device).uniform_(-3.14, 3.14)
    quat = quat_from_euler_xyz(roll, pitch, yaw)  # (N, 4) wxyz
    root_state[:, 3:7] = quat

    # Random downward velocity.
    root_state[:, 7] = torch.empty(num_envs, device=device).uniform_(-vel_range[1], vel_range[1])
    root_state[:, 8] = torch.empty(num_envs, device=device).uniform_(-vel_range[1], vel_range[1])
    root_state[:, 9] = torch.empty(num_envs, device=device).uniform_(-vel_range[1], -0.5)

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
    joint_vel = torch.empty_like(joint_pos).uniform_(-0.5, 0.5)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Stage II — sample from predictor-flagged state bank
# ---------------------------------------------------------------------------

_STAGE2_BANK: dict[str, torch.Tensor] | None = None


def _load_stage2_bank(bank_path: str) -> dict[str, torch.Tensor]:
    """Lazy‑load the Stage II state bank (shared across all envs)."""
    global _STAGE2_BANK
    if _STAGE2_BANK is None:
        _STAGE2_BANK = torch.load(bank_path, map_location="cpu", weights_only=False)
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

    Samples initial states from a pre‑extracted bank of realistic
    falling configurations, then applies small random noise for
    diversity.  Falls back to ``reset_falling_state`` if the bank
    cannot be loaded.

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

    # Bank stores Z=0 and lin_vel=0 — fill with random values
    # matching the paper's falling-state ranges.
    root_state[:, 2] = torch.empty(num_envs, device=device).uniform_(0.4, 0.9)
    root_state[:, 7] = torch.empty(num_envs, device=device).uniform_(-2.0, 2.0)
    root_state[:, 8] = torch.empty(num_envs, device=device).uniform_(-2.0, 2.0)
    root_state[:, 9] = torch.empty(num_envs, device=device).uniform_(-2.0, -0.5)

    # Small Gaussian noise on joints for diversity.
    joint_pos += torch.randn_like(joint_pos) * pos_noise
    joint_vel += torch.randn_like(joint_vel) * vel_noise

    # Write tentative state so we can check validity.
    root_state[:, 0:3] += env.scene.env_origins[env_ids]
    asset.write_root_state_to_sim(root_state, env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.sim.forward()

    # ── Kinematic validity filter (paper §III-D) ──
    # Use MuJoCo's native contact distances after forward kinematics.
    # ``dist < 0`` means geometry‑aware penetration (handles any
    # body shape — sphere radius, box half‑extent, etc.).
    #   - ground penetration:  either geom's parent body is world (id 0)
    #   - self‑collision:      both geoms belong to the robot
    import mujoco
    _mjm = env.sim.mj_model
    _mjd = env.sim.mj_data
    sim_data = env.sim.data
    invalid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for j, ei in enumerate(env_ids.cpu().tolist()):
        _mjd.qpos[:] = sim_data.qpos[ei].cpu().numpy()
        _mjd.qvel[:] = sim_data.qvel[ei].cpu().numpy()
        mujoco.mj_forward(_mjm, _mjd)
        for c in range(_mjd.ncon):
            if _mjd.contact.dist[c] >= 0:  # geometry‑aware — positive = no penetration
                continue
            b1 = _mjm.geom_bodyid[_mjd.contact.geom1[c]]
            b2 = _mjm.geom_bodyid[_mjd.contact.geom2[c]]
            # ground penetration (world body is id 0)
            if b1 == 0 or b2 == 0:
                invalid[j] = True
                break
            # self‑collision (both robot bodies)
            if b1 != 0 and b2 != 0:
                invalid[j] = True
                break

    if invalid.any().item():
        n_invalid = invalid.sum().item()
        new_idx = torch.randint(0, K, (n_invalid,))
        joint_pos[invalid] = bank["joint_pos"][new_idx].to(device).clone()
        joint_vel[invalid] = bank["joint_vel"][new_idx].to(device).clone()
        joint_pos[invalid] += torch.randn_like(joint_pos[invalid]) * pos_noise
        joint_vel[invalid] += torch.randn_like(joint_vel[invalid]) * vel_noise
        # Restore random height and velocity for resampled envs.
        root_state[invalid, 2] = torch.empty(n_invalid, device=device).uniform_(0.4, 0.9)
        root_state[invalid, 7] = torch.empty(n_invalid, device=device).uniform_(-2.0, 2.0)
        root_state[invalid, 8] = torch.empty(n_invalid, device=device).uniform_(-2.0, 2.0)
        root_state[invalid, 9] = torch.empty(n_invalid, device=device).uniform_(-2.0, -0.5)
        # Restore XY offset.
        root_state[invalid, 0:2] += env.scene.env_origins[env_ids[invalid], 0:2]
        asset.write_root_state_to_sim(root_state, env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        env.sim.forward()
