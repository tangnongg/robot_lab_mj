#!/usr/bin/env python3
"""Record and plot SafeFall body/joint dynamics and MuJoCo forces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_src = Path(__file__).resolve().parents[4]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import robot_lab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _contact_force(env: ManagerBasedRlEnv) -> np.ndarray:
    data = env.scene["ground_contact"].data
    if data.force is not None:
        return _to_numpy(torch.linalg.vector_norm(data.force[0], dim=-1))
    if data.found is not None:
        return _to_numpy(data.found[0].float())
    return np.zeros(1, dtype=np.float32)


def _record_step(env: ManagerBasedRlEnv) -> dict[str, np.ndarray]:
    asset = env.scene["robot"]
    sim_data = env.sim.data
    body_ids = asset.data.indexing.body_ids.long()
    joint_ids = asset.data.indexing.joint_v_adr.long()

    body_lin = asset.data.body_link_lin_vel_w[0]
    body_ang = asset.data.body_link_ang_vel_w[0]
    qvel = sim_data.qvel[0, joint_ids]
    qacc = sim_data.qacc[0, joint_ids]
    cfrc_int = sim_data.cfrc_int[0, body_ids]
    cfrc_ext = sim_data.cfrc_ext[0, body_ids]
    head_local_ids, head_names = asset.find_geoms(("head_collision",), preserve_order=True)
    if head_names != ["head_collision"]:
        raise RuntimeError("SafeFall dynamics report requires head_collision")
    head_geom_id = int(asset.indexing.geom_ids[head_local_ids[0]])
    head_radius = float(env.sim.mj_model.geom_size[head_geom_id, 0])
    head_clearance = (
        sim_data.geom_xpos[0, head_geom_id, 2]
        - head_radius
        - env.scene.env_origins[0, 2]
    )
    contact_names = env.scene["ground_contact"].primary_names
    head_contact_index = contact_names.index("head_collision")
    ground_force = env.scene["ground_contact"].data.force
    head_force = (
        torch.linalg.vector_norm(ground_force[0, head_contact_index], dim=-1)
        if ground_force is not None else torch.zeros((), device=env.device)
    )
    limb_ids = [i for i, name in enumerate(asset.body_names) if name.startswith(("left_", "right_"))]
    limb_global_ids = body_ids[limb_ids].cpu().numpy()
    limb_masses = torch.as_tensor(env.sim.mj_model.body_mass[limb_global_ids], device=env.device)
    limb_height = asset.data.body_com_pos_w[0, limb_ids, 2] - env.scene.env_origins[0, 2]
    limb_com_height = torch.sum(limb_masses * limb_height) / limb_masses.sum().clamp(min=1.0e-6)
    torso_id = asset.body_names.index("torso_link")
    knee_ids = [asset.body_names.index(name) for name in ("left_knee_link", "right_knee_link")]
    ankle_ids = [
        asset.body_names.index(name)
        for name in ("left_ankle_pitch_link", "right_ankle_pitch_link")
    ]
    body_com = asset.data.body_com_pos_w[0]
    torso_com = body_com[torso_id]
    leg_fold_distance = torch.linalg.vector_norm(
        body_com[torch.as_tensor(knee_ids + ankle_ids, device=env.device)] - torso_com,
        dim=-1,
    )
    knee_joint_ids = [
        asset.joint_names.index(name)
        for name in ("left_knee_joint", "right_knee_joint")
    ]
    return {
        "body_lin_vel": _to_numpy(body_lin),
        "body_ang_vel": _to_numpy(body_ang),
        "joint_vel": _to_numpy(qvel),
        "joint_acc": _to_numpy(qacc),
        "cfrc_int": _to_numpy(cfrc_int),
        "cfrc_ext": _to_numpy(cfrc_ext),
        "qfrc_constraint": _to_numpy(sim_data.qfrc_constraint[0, joint_ids]),
        "qfrc_actuator": _to_numpy(sim_data.qfrc_actuator[0, joint_ids]),
        "contact_force": _contact_force(env),
        "root_height": _to_numpy(asset.data.root_link_pos_w[0, 2:3]),
        "head_clearance": _to_numpy(head_clearance.unsqueeze(0)),
        "head_contact_force": _to_numpy(head_force.unsqueeze(0)),
        "limb_com_height": _to_numpy(limb_com_height.unsqueeze(0)),
        "leg_fold_distance": _to_numpy(leg_fold_distance),
        "knee_joint_pos": _to_numpy(asset.data.joint_pos[0, knee_joint_ids]),
    }


def _metric_arrays(data: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    body_lin = data["body_lin_vel"]
    body_ang = data["body_ang_vel"]
    joint_vel = data["joint_vel"]
    joint_acc = data["joint_acc"]
    body_lin_speed = np.linalg.norm(body_lin, axis=-1)
    body_ang_speed = np.linalg.norm(body_ang, axis=-1)
    body_lin_acc = np.gradient(body_lin, dt, axis=0)
    body_ang_acc = np.gradient(body_ang, dt, axis=0)
    return {
        "body_lin_speed": body_lin_speed,
        "body_ang_speed": body_ang_speed,
        "body_lin_acc": np.linalg.norm(body_lin_acc, axis=-1),
        "body_ang_acc": np.linalg.norm(body_ang_acc, axis=-1),
        "joint_speed": np.abs(joint_vel),
        "joint_acc_abs": np.abs(joint_acc),
        "cfrc_int_force": np.linalg.norm(data["cfrc_int"][..., 3:6], axis=-1),
        "cfrc_int_torque": np.linalg.norm(data["cfrc_int"][..., :3], axis=-1),
        "cfrc_ext_force": np.linalg.norm(data["cfrc_ext"][..., 3:6], axis=-1),
        "cfrc_ext_torque": np.linalg.norm(data["cfrc_ext"][..., :3], axis=-1),
        "qfrc_constraint_abs": np.abs(data["qfrc_constraint"]),
        "qfrc_actuator_abs": np.abs(data["qfrc_actuator"]),
        "contact_force": data["contact_force"],
        "root_height": data["root_height"].squeeze(-1),
        "head_clearance": data["head_clearance"].squeeze(-1),
        "head_contact_force": data["head_contact_force"].squeeze(-1),
        "limb_com_height": data["limb_com_height"].squeeze(-1),
        "leg_fold_distance": data["leg_fold_distance"],
        "knee_joint_pos": data["knee_joint_pos"],
    }


def _stack(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack([row[name] for row in records], axis=0) for name in records[0]}


def _settling_report(metrics: dict[str, np.ndarray], dt: float) -> dict[str, object]:
    force = metrics["contact_force"].max(axis=1)
    contact = np.flatnonzero(force > 20.0)
    contact_step = int(contact[0]) if contact.size else None
    root_lin = metrics["body_lin_speed"][:, 0]
    root_ang = metrics["body_ang_speed"][:, 0]
    joint_rms = np.sqrt(np.mean(metrics["joint_speed"] ** 2, axis=1))
    body_lin_max = metrics["body_lin_speed"].max(axis=1)
    body_ang_max = metrics["body_ang_speed"].max(axis=1)

    if contact_step is None:
        return {"contact_step": None, "contact_time_s": None}
    pre = slice(0, max(contact_step, 1))
    limits = {
        "root_lin_speed": max(float(np.max(root_lin[pre])) * 0.1, 0.15),
        "root_ang_speed": max(float(np.max(root_ang[pre])) * 0.1, 0.5),
        "joint_vel_rms": max(float(np.max(joint_rms[pre])) * 0.1, 0.5),
    }
    quiet = (
        (root_lin <= limits["root_lin_speed"])
        & (root_ang <= limits["root_ang_speed"])
        & (joint_rms <= limits["joint_vel_rms"])
    )
    stable_steps = 5
    quiet_run = np.convolve(quiet.astype(np.int32), np.ones(stable_steps, dtype=np.int32), mode="same")
    settled = np.flatnonzero((quiet_run >= stable_steps) & (np.arange(len(quiet)) >= contact_step))
    settled_step = int(settled[0]) if settled.size else None
    tail = slice(-min(20, len(root_lin)), None)
    # The training phase locks after 30 physics substeps.  A policy step is
    # 0.02 s here, so this eight-step slice covers the impact-only reward.
    impact = slice(contact_step, min(contact_step + 8, len(root_lin)))
    return {
        "contact_step": contact_step,
        "contact_time_s": contact_step * dt,
        "settled_step_5_policy_steps": settled_step,
        "settled_time_s": None if settled_step is None else settled_step * dt,
        "limits": limits,
        "precontact_peaks": {
            "root_lin_speed": float(np.max(root_lin[pre])),
            "root_ang_speed": float(np.max(root_ang[pre])),
            "joint_vel_rms": float(np.max(joint_rms[pre])),
        },
        "tail20_mean": {
            "root_lin_speed": float(np.mean(root_lin[tail])),
            "root_ang_speed": float(np.mean(root_ang[tail])),
            "joint_vel_rms": float(np.mean(joint_rms[tail])),
            "body_lin_speed_max": float(np.mean(body_lin_max[tail])),
            "body_ang_speed_max": float(np.mean(body_ang_max[tail])),
            "joint_acc_rms": float(np.mean(np.sqrt(np.mean(metrics["joint_acc_abs"][:, :] ** 2, axis=1))[tail])),
            "head_clearance_m": float(np.mean(metrics["head_clearance"][tail])),
            "head_contact_force_n": float(np.mean(metrics["head_contact_force"][tail])),
            "limb_com_height_m": float(np.mean(metrics["limb_com_height"][tail])),
            "minimum_knee_torso_distance_m": float(np.min(metrics["leg_fold_distance"][tail, :2])),
            "minimum_ankle_torso_distance_m": float(np.min(metrics["leg_fold_distance"][tail, 2:])),
            "peak_knee_angle_rad": float(np.max(metrics["knee_joint_pos"][tail])),
        },
        "posture_safety": {
            "minimum_head_clearance_m": float(np.min(metrics["head_clearance"])),
            "peak_head_contact_force_n": float(np.max(metrics["head_contact_force"])),
            "impact_minimum_knee_torso_distance_m": float(
                np.min(metrics["leg_fold_distance"][impact, :2])
            ),
            "impact_minimum_ankle_torso_distance_m": float(
                np.min(metrics["leg_fold_distance"][impact, 2:])
            ),
            "minimum_knee_torso_distance_m": float(np.min(metrics["leg_fold_distance"][:, :2])),
            "minimum_ankle_torso_distance_m": float(np.min(metrics["leg_fold_distance"][:, 2:])),
            "peak_knee_angle_rad": float(np.max(metrics["knee_joint_pos"])),
        },
    }


def _heatmap(ax, values: np.ndarray, times: np.ndarray, labels: list[str], title: str, cmap: str = "magma") -> None:
    image = ax.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(times[0], times[-1], -0.5, len(labels) - 0.5),
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("index")
    if len(labels) <= 40:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
    plt.colorbar(image, ax=ax, shrink=0.8)


def _plot_all(metrics: dict[str, np.ndarray], names: dict[str, list[str]], dt: float, output_dir: Path, report: dict[str, object]) -> None:
    times = np.arange(metrics["body_lin_speed"].shape[0]) * dt
    contact_step = report.get("contact_step")
    contact_time = None if contact_step is None else float(contact_step) * dt

    root_lin = metrics["body_lin_speed"][:, 0]
    root_ang = metrics["body_ang_speed"][:, 0]
    joint_rms = np.sqrt(np.mean(metrics["joint_speed"] ** 2, axis=1))
    joint_acc_rms = np.sqrt(np.mean(metrics["joint_acc_abs"] ** 2, axis=1))
    contact = metrics["contact_force"].max(axis=1)
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    curves = [
        (root_lin, "root linear speed (m/s)"),
        (root_ang, "root angular speed (rad/s)"),
        (joint_rms, "joint velocity RMS (rad/s)"),
        (joint_acc_rms, "joint acceleration RMS (rad/s²)"),
        (contact, "maximum terrain contact force (N)"),
        (metrics["root_height"], "root height (m)"),
        (metrics["head_clearance"], "head terrain clearance (m)"),
        (metrics["limb_com_height"], "mass-weighted limb COM height (m)"),
    ]
    for ax, (value, title) in zip(axes.flat, curves, strict=True):
        ax.plot(times, value, linewidth=1.2)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if contact_time is not None:
            ax.axvline(contact_time, color="tab:red", linestyle="--", label="first contact")
            ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    _heatmap(axes[0, 0], metrics["body_lin_speed"], times, names["body"], "body linear speed", "viridis")
    _heatmap(axes[0, 1], metrics["body_ang_speed"], times, names["body"], "body angular speed", "viridis")
    _heatmap(axes[1, 0], metrics["body_lin_acc"], times, names["body"], "body linear acceleration", "inferno")
    _heatmap(axes[1, 1], metrics["body_ang_acc"], times, names["body"], "body angular acceleration", "inferno")
    fig.tight_layout()
    fig.savefig(output_dir / "body_kinematics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    _heatmap(axes[0], metrics["joint_speed"], times, names["joint"], "joint speed", "viridis")
    _heatmap(axes[1], metrics["joint_acc_abs"], times, names["joint"], "joint acceleration", "inferno")
    fig.tight_layout()
    fig.savefig(output_dir / "joint_kinematics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    force_plots = [
        (metrics["cfrc_int_force"], names["body"], "cfrc_int force"),
        (metrics["cfrc_int_torque"], names["body"], "cfrc_int torque"),
        (metrics["cfrc_ext_force"], names["body"], "cfrc_ext force"),
        (metrics["cfrc_ext_torque"], names["body"], "cfrc_ext torque"),
        (metrics["qfrc_constraint_abs"], names["joint"], "|qfrc_constraint|"),
        (metrics["qfrc_actuator_abs"], names["joint"], "|qfrc_actuator|"),
    ]
    for ax, (value, labels, title) in zip(axes.flat, force_plots, strict=True):
        _heatmap(ax, value, times, labels, title, "magma")
    fig.tight_layout()
    fig.savefig(output_dir / "forces.png", dpi=180)
    plt.close(fig)


def run(checkpoint: str | Path, output_dir: str | Path, steps: int, device: str) -> None:
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = "Mjlab-SafeFall-G1"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        steps * env_cfg.decimation * env_cfg.sim.mujoco.timestep + 0.1,
    )
    base_env = ManagerBasedRlEnv(env_cfg, device=device)
    agent_cfg = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    records: list[dict[str, np.ndarray]] = []
    try:
        for _ in range(steps):
            with torch.inference_mode():
                env.step(policy(env.get_observations()))
            records.append(_record_step(base_env))
    finally:
        env.close()
    raw = _stack(records)
    dt = base_env.step_dt
    metrics = _metric_arrays(raw, dt)
    names = {
        "body": list(base_env.scene["robot"].body_names),
        "joint": list(base_env.scene["robot"].joint_names),
    }
    report = _settling_report(metrics, dt)
    arrays = dict(raw)
    arrays.update({f"metric_{name}": value for name, value in metrics.items()})
    np.savez_compressed(output_dir / "dynamics.npz", **arrays)
    (output_dir / "names.json").write_text(json.dumps(names, indent=2))
    (output_dir / "settling_report.json").write_text(json.dumps(report, indent=2))
    _plot_all(metrics, names, dt, output_dir, report)
    print(json.dumps({"output_dir": str(output_dir), **report}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.checkpoint, args.output_dir, args.steps, args.device)


if __name__ == "__main__":
    main()
