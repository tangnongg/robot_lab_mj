#!/usr/bin/env python3
"""Calibrate SafeFall reward weights (w_c, w_j, w_e) by measuring
per‑term magnitude over random falling episodes.

Paper Eq.2:  r_impact = w_c·r_contact + w_j·r_joint + w_e·r_torque

Run this BEFORE training to set reasonable weight orders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_src = Path(__file__).resolve().parents[4]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

def calibrate(
    task_id: str = "Mjlab-SafeFall-G1",
    num_envs: int = 256,
    num_episodes: int = 5,
    device: str = "cpu",
    seed: int = 42,
    target_step_cost: float = 1.0,
) -> dict[str, float]:
    """Run falling episodes and report per‑term reward statistics."""

    torch.manual_seed(seed)

    cfg = load_env_cfg(task_id, play=False)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg, device=device)

    # Find term indices via private internals (stable within a major version).
    rm = env.reward_manager
    names = rm._term_names       # list[str] — order matches _term_cfgs / _step_reward
    idx_contact = names.index("r_contact")
    idx_joint   = names.index("r_joint")
    idx_torque  = names.index("r_torque")
    w_contact = abs(rm._term_cfgs[idx_contact].weight)
    w_joint   = abs(rm._term_cfgs[idx_joint].weight)
    w_torque  = abs(rm._term_cfgs[idx_torque].weight)

    r_contact_vals: list[float] = []
    r_joint_vals: list[float] = []
    r_torque_vals: list[float] = []

    # Run episodes to completion (up to 40 steps).  Most steps the
    # robot is airborne so r_impact terms are zero.  Capture only
    # non‑zero per‑env values so the statistics reflect the impact
    # phase — not the idle falling phase.
    for _ep in range(num_episodes):
        env.reset()
        for _step in range(40):
            action = torch.randn(num_envs, 29, device=device)
            env.step(action)

            sr = rm._step_reward  # (B, n_terms)
            # Impact functions return non-negative costs and their configured
            # weights are negative. Convert the weighted reward rate back to a
            # positive raw cost for an easier-to-read calibration report.
            raw_c = -sr[:, idx_contact] / w_contact  # (B,)
            raw_j = -sr[:, idx_joint] / w_joint
            raw_e = -sr[:, idx_torque] / w_torque

            # Only keep non‑zero values (active impact phase).
            mask_c = raw_c.abs() > 0
            mask_j = raw_j.abs() > 0
            mask_e = raw_e.abs() > 0

            if mask_c.any():
                for v in raw_c[mask_c].cpu().tolist():
                    r_contact_vals.append(v)
            if mask_j.any():
                for v in raw_j[mask_j].cpu().tolist():
                    r_joint_vals.append(v)
            if mask_e.any():
                for v in raw_e[mask_e].cpu().tolist():
                    r_torque_vals.append(v)

    step_dt = env.step_dt
    env.close()

    rc = np.array(r_contact_vals)
    rj = np.array(r_joint_vals)
    rt = np.array(r_torque_vals)

    stats = {}
    for name, arr in [("r_contact", rc), ("r_joint", rj), ("r_torque", rt)]:
        if arr.size == 0:
            raise RuntimeError(f"{name} never became non-zero during calibration")
        stats[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "abs_mean": float(np.abs(arr).mean()),
        }

    # Choose each weight so a typical non-zero impact sample contributes
    # target_step_cost after RewardManager's dt scaling.
    mag_c = stats["r_contact"]["abs_mean"]
    mag_j = stats["r_joint"]["abs_mean"]
    mag_e = stats["r_torque"]["abs_mean"]
    weights = {
        "w_c": target_step_cost / (step_dt * max(mag_c, 1e-9)),
        "w_j": target_step_cost / (step_dt * max(mag_j, 1e-9)),
        "w_e": target_step_cost / (step_dt * max(mag_e, 1e-9)),
    }

    # ── Print report ──
    print("=" * 64)
    print("  SafeFall reward weight calibration")
    print(f"  task={task_id}  envs={num_envs}  episodes={num_episodes}")
    print("=" * 64)
    print()
    print("  Raw per‑step statistics (unweighted):")
    print(f"  {'':<14s} {'mean':>10s} {'std':>10s} {'abs_mean':>10s} "
          f"{'min':>10s} {'max':>10s}")
    for name, s in stats.items():
        print(f"  {name:<14s} {s['mean']:10.4f} {s['std']:10.4f} "
              f"{s['abs_mean']:10.4f} {s['min']:10.4f} {s['max']:10.4f}")
    print()
    print(f"  Calibrated weights (typical contribution ~{target_step_cost:g}/step):")
    for k, v in weights.items():
        print(f"    {k} = {v:.1e}")
    print()
    print("  Suggested env_cfgs.py snippet:")
    print(f'    "r_contact": RewardTermCfg(func=mdp.ContactForcePenalty, '
          f'weight={-weights["w_c"]:.2e}),')
    print(f'    "r_joint":   RewardTermCfg(func=mdp.reward_joint_reaction, '
          f'weight={-weights["w_j"]:.2e}),')
    print(f'    "r_torque":  RewardTermCfg(func=mdp.reward_joint_torques, '
          f'weight={-weights["w_e"]:.2e}),')
    print("=" * 64)

    return weights


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", default="Mjlab-SafeFall-G1")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-step-cost", type=float, default=1.0)
    args = p.parse_args()
    calibrate(
        task_id=args.task_id,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        device=args.device,
        seed=args.seed,
        target_step_cost=args.target_step_cost,
    )
