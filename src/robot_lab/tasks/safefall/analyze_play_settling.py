#!/usr/bin/env python3
"""Analyze post-impact settling metrics from a SafeFall play rollout."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import torch

_src = Path(__file__).resolve().parents[4]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import robot_lab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def _scalar(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _ground_force(env: ManagerBasedRlEnv) -> torch.Tensor:
    sensor = env.scene["ground_contact"]
    data = sensor.data
    if data.force_history is not None:
        return torch.linalg.vector_norm(data.force_history, dim=-1).amax(dim=(1, 2))
    if data.force is not None:
        return torch.linalg.vector_norm(data.force, dim=-1).amax(dim=1)
    if data.found is not None:
        return data.found.amax(dim=1).float()
    return torch.zeros(env.num_envs, device=env.device)


def _motion_metrics(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    asset = env.scene["robot"]
    qvel = env.sim.data.qvel
    root_qvel = qvel[:, asset.data.indexing.free_joint_v_adr.long()]
    joint_qvel = qvel[:, asset.data.indexing.joint_v_adr.long()]
    joint_acc = env.sim.data.qacc[:, asset.data.indexing.joint_v_adr.long()]
    return {
        "ground_force": _ground_force(env),
        "root_lin_speed": torch.linalg.vector_norm(root_qvel[:, :3], dim=-1),
        "root_ang_speed": torch.linalg.vector_norm(root_qvel[:, 3:6], dim=-1),
        "joint_vel_rms": torch.sqrt(torch.mean(torch.square(joint_qvel), dim=-1)),
        "joint_acc_rms": torch.sqrt(torch.mean(torch.square(joint_acc), dim=-1)),
    }


def _first_settled_step(
    rows: list[dict[str, float]],
    *,
    settle_ratio: float,
    lin_floor: float,
    ang_floor: float,
    joint_floor: float,
    settle_steps: int,
) -> tuple[int | None, dict[str, float]]:
    contact_step = next((i for i, row in enumerate(rows) if row["ground_force"] > 20.0), None)
    if contact_step is None:
        return None, {}
    pre_rows = rows[: max(contact_step, 1)]
    limits = {
        "root_lin_speed": max(max(r["root_lin_speed"] for r in pre_rows) * settle_ratio, lin_floor),
        "root_ang_speed": max(max(r["root_ang_speed"] for r in pre_rows) * settle_ratio, ang_floor),
        "joint_vel_rms": max(max(r["joint_vel_rms"] for r in pre_rows) * settle_ratio, joint_floor),
    }
    quiet_count = 0
    for i, row in enumerate(rows[contact_step:], start=contact_step):
        quiet = all(row[name] <= limit for name, limit in limits.items())
        quiet_count = quiet_count + 1 if quiet else 0
        if quiet_count >= settle_steps:
            return i, limits
    return None, limits


def analyze(
    checkpoint: str | Path,
    *,
    steps: int = 200,
    device: str = "cuda:0",
    csv_path: str | Path | None = None,
) -> None:
    checkpoint = Path(checkpoint).resolve()
    task_id = "Mjlab-SafeFall-G1"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, steps * env_cfg.decimation * env_cfg.sim.mujoco.timestep)

    base_env = ManagerBasedRlEnv(env_cfg, device=device)
    agent_cfg = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    rows: list[dict[str, float]] = []
    try:
        for step in range(steps):
            with torch.inference_mode():
                env.step(policy(env.get_observations()))
            metrics = _motion_metrics(base_env)
            row = {"step": float(step), "time_s": step * base_env.step_dt}
            row.update({name: _scalar(value[0]) for name, value in metrics.items()})
            rows.append(row)
    finally:
        env.close()

    if csv_path is None:
        csv_path = checkpoint.parent / "play_settling_metrics.csv"
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    settled_step, limits = _first_settled_step(
        rows,
        settle_ratio=0.1,
        lin_floor=0.15,
        ang_floor=0.5,
        joint_floor=0.5,
        settle_steps=5,
    )
    contact_step = next((i for i, row in enumerate(rows) if row["ground_force"] > 20.0), None)
    print(f"csv={csv_path}")
    print(f"contact_step={contact_step}")
    print(f"settled_step_5_policy_steps={settled_step}")
    print(f"limits={limits}")
    for name in ("root_lin_speed", "root_ang_speed", "joint_vel_rms", "joint_acc_rms"):
        values = torch.tensor([row[name] for row in rows])
        print(
            f"{name}: max={values.max().item():.4f} "
            f"p50={values.quantile(0.50).item():.4f} "
            f"p90={values.quantile(0.90).item():.4f} "
            f"tail20_mean={values[-20:].mean().item():.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv-path")
    args = parser.parse_args()
    analyze(args.checkpoint, steps=args.steps, device=args.device, csv_path=args.csv_path)


if __name__ == "__main__":
    main()
