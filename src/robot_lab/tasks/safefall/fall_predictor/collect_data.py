#!/usr/bin/env python3
"""Collect falling trajectories for fall predictor training — paper §III-B.

Runs a nominal locomotion policy under perturbation to induce falls,
recording proprioceptive observations up to ground impact.  Supports
multi-environment parallel collection for throughput.

Real-time progress displays per-trajectory detail plus periodic summaries.
After collection, a REPORT.txt is written to the output directory.

Paper Table I — 6 failure factors implemented:
  1. Sensor noise      — scale observation noise 2-10×
  2. External force     — velocity perturbation to torso
  3. Foot slip          — velocity kick to stance foot body
  4. Foot trip          — rough terrain (--rough flag)
  5. System delay       — FIFO pipeline delay [0, 200] ms
  6. Dynamic mismatch   — PD gain randomization + CoM offset at reset

Usage:
    # Flat terrain, single env
    python -m robot_lab.tasks.safefall.fall_predictor.collect_data \\
        --output data/fall_trajs --num-trajs 5000 \\
        --policy-checkpoint logs/rsl_rl/.../model_1999.pt \\
        --policy-task Mjlab-Velocity-Flat-Unitree-G1

    # 64 parallel envs for throughput
    python -m robot_lab.tasks.safefall.fall_predictor.collect_data \\
        --output data/fall_trajs --num-trajs 5000 --num-envs 64 \\
        --policy-checkpoint logs/rsl_rl/.../model_1999.pt

    # Rough terrain (foot trip factor)
    python -m robot_lab.tasks.safefall.fall_predictor.collect_data \\
        --output data/fall_trajs_rough --num-trajs 5000 \\
        --policy-checkpoint logs/rsl_rl/.../model_1999.pt \\
        --policy-task Mjlab-Velocity-Flat-Unitree-G1 --rough

    # With viewer (single env only)
    python -m robot_lab.tasks.safefall.fall_predictor.collect_data \\
        --output data/fall_trajs --num-trajs 500 \\
        --policy-checkpoint logs/rsl_rl/.../model_1999.pt --viewer
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# Ensure robot_lab is importable.
_src = Path(__file__).resolve().parents[4]  # robot_lab_mj/src
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner

from robot_lab.tasks.safefall.fall_predictor.dataset import (
    TrajectoryWriter,
    _T2_OFFSET_STEPS,
)
from robot_lab.tasks.safefall.fall_predictor.model import (
    INPUT_DIM, extract_predictor_input,
)

logging.getLogger("mjlab").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

FACTOR_NAMES = {
    0: "sensor_noise",
    1: "external_force",
    2: "foot_slip",
    3: "foot_trip",
    4: "system_delay",
    5: "dynamic_mismatch",
}


@dataclass
class PerturbConfig:
    """Configuration for a single perturbation episode."""

    sensor_noise_scale: float = 1.0                 # 0 — sensor noise
    external_vel: tuple[float, float] = (0.0, 0.0)  # 1 — external force
    push_at_step: int = 50
    foot_slip_vel: float = 0.0                      # 2 — foot slip
    slip_at_step: int = 60
    delay_steps: int = 0                             # 4 — system delay
    pd_gain_scale: tuple[float, float] = (1.0, 1.0) # 5 — dynamic mismatch
    com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def active_factors(self) -> list[str]:
        names: list[str] = []
        if self.sensor_noise_scale > 1.0:
            names.append(f"sensor_noise({self.sensor_noise_scale:.1f}×)")
        if self.external_vel != (0.0, 0.0):
            names.append(f"external_force(vx={self.external_vel[0]:.1f},"
                         f"vy={self.external_vel[1]:.1f}m/s)")
        if self.foot_slip_vel != 0.0:
            names.append(f"foot_slip({self.foot_slip_vel:.1f}m/s)")
        if self.delay_steps > 0:
            names.append(f"delay({self.delay_steps * 20}ms)")
        kp_act = self.pd_gain_scale != (1.0, 1.0)
        com_act = self.com_offset != (0.0, 0.0, 0.0)
        if kp_act or com_act:
            parts: list[str] = []
            if kp_act:
                parts.append(f"kp{self.pd_gain_scale[0]:.1f}/"
                             f"kd{self.pd_gain_scale[1]:.1f}")
            if com_act:
                parts.append(f"CoM({self.com_offset[0]:.2f},"
                             f"{self.com_offset[1]:.2f})")
            names.append("dyn_mismatch(" + ",".join(parts) + ")")
        return names


def _random_perturb(rng: np.random.Generator) -> PerturbConfig:
    """Sample a random perturbation configuration (1-3 factors)."""
    cfg = PerturbConfig()
    n_factors = rng.integers(1, 4)
    factors = rng.choice([0, 1, 2, 4, 5], size=n_factors, replace=False)

    for f in factors:
        if f == 0:
            cfg.sensor_noise_scale = float(rng.uniform(2.0, 10.0))
        elif f == 1:
            cfg.external_vel = (float(rng.uniform(-2.0, 2.0)),
                                float(rng.uniform(-1.0, 1.0)))
            cfg.push_at_step = int(rng.integers(20, 80))
        elif f == 2:
            cfg.foot_slip_vel = float(rng.uniform(-1.5, 1.5))
            cfg.slip_at_step = int(rng.integers(20, 80))
        elif f == 4:
            cfg.delay_steps = int(rng.integers(1, 11))
        elif f == 5:
            cfg.pd_gain_scale = (
                float(np.exp(rng.uniform(np.log(0.7), np.log(1.5)))),
                float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))),
            )
            cfg.com_offset = (rng.uniform(-0.05, 0.05),
                              rng.uniform(-0.05, 0.05),
                              rng.uniform(-0.01, 0.01))
    return cfg


# ---------------------------------------------------------------------------
# Statistics accumulator
# ---------------------------------------------------------------------------

@dataclass
class CollectStats:
    """Running statistics across all episodes."""
    total_episodes: int = 0
    saved_trajs: int = 0
    rejected_no_fall: int = 0
    rejected_too_short: int = 0

    traj_lengths: list[int] = field(default_factory=list)
    init_heights: list[float] = field(default_factory=list)
    final_heights: list[float] = field(default_factory=list)
    impact_velocities: list[float] = field(default_factory=list)

    factor_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int))

    start_time: float = 0.0
    label_safe: int = 0
    label_ambiguous: int = 0
    label_falling: int = 0

    def record_saved(self, traj_len: int, init_h: float, final_h: float,
                     impact_vel: float, factors: list[str],
                     n_safe: int, n_amb: int, n_fall: int) -> None:
        self.saved_trajs += 1
        self.traj_lengths.append(traj_len)
        self.init_heights.append(init_h)
        self.final_heights.append(final_h)
        self.impact_velocities.append(impact_vel)
        for f in factors:
            self.factor_counts[f.split("(")[0]] += 1
        self.label_safe += n_safe
        self.label_ambiguous += n_amb
        self.label_falling += n_fall

    def record_rejected(self, had_fall: bool, too_short: bool) -> None:
        if too_short:
            self.rejected_too_short += 1
        elif not had_fall:
            self.rejected_no_fall += 1

    @property
    def rejection_rate(self) -> float:
        total = self.rejected_no_fall + self.rejected_too_short
        attempted = self.saved_trajs + total
        return total / max(attempted, 1)

    @property
    def avg_traj_len(self) -> float:
        return float(np.mean(self.traj_lengths)) if self.traj_lengths else 0.0

    @property
    def avg_init_height(self) -> float:
        return float(np.mean(self.init_heights)) if self.init_heights else 0.0

    @property
    def avg_final_height(self) -> float:
        return float(np.mean(self.final_heights)) if self.final_heights else 0.0

    @property
    def avg_impact_vel(self) -> float:
        return float(np.mean(self.impact_velocities)) if self.impact_velocities else 0.0

    @property
    def label_summary(self) -> str:
        total = self.label_safe + self.label_ambiguous + self.label_falling
        if total == 0:
            return "N/A"
        return (f"safe={self.label_safe/total:.1%}  "
                f"ambig={self.label_ambiguous/total:.1%}  "
                f"falling={self.label_falling/total:.1%}")


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

_TERM_WIDTH = 100
_HR = "─" * _TERM_WIDTH


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_header(args_dict: dict, output_dir: str) -> None:
    print(f"\n{_HR}\n  SafeFall Data Collection  |  paper §III-B\n{_HR}")
    for k, v in args_dict.items():
        print(f"  {k:<22s}: {v}")
    print(f"  {'output':<22s}: {output_dir}")
    print(f"{_HR}\n")


def _print_progress(stats: CollectStats, target: int, t_start: float):
    elapsed = time.time() - t_start
    done = stats.saved_trajs
    pct = done / max(target, 1) * 100
    rate = done / max(elapsed, 1)
    eta = (target - done) / max(rate, 1e-9)
    bar_w = 30
    filled = int(bar_w * done / max(target, 1))
    bar = "█" * filled + "░" * (bar_w - filled)
    line = (
        f"\r  [{bar}] {done:5d}/{target} ({pct:5.1f}%)  "
        f"|  {rate:.2f} traj/s  |  ETA {_fmt_duration(eta)}  "
        f"|  avg_len={stats.avg_traj_len:.0f}  "
        f"|  rej={stats.rejection_rate:.0%}  "
        f"|  {_fmt_duration(elapsed)} elapsed"
    )
    print(line.ljust(_TERM_WIDTH), end="", flush=True)


def _print_checkpoint(stats: CollectStats, target: int, t_start: float):
    elapsed = time.time() - t_start
    print(f"\n  ── Checkpoint @ {stats.saved_trajs}/{target} "
          f"({_fmt_duration(elapsed)} elapsed) " + "─" * 40)
    print(f"  Trajectory lengths : μ={stats.avg_traj_len:.0f}  "
          f"min={min(stats.traj_lengths) if stats.traj_lengths else 0}  "
          f"max={max(stats.traj_lengths) if stats.traj_lengths else 0}")
    print(f"  Init height (m)    : μ={stats.avg_init_height:.3f}")
    print(f"  Final height (m)   : μ={stats.avg_final_height:.4f}")
    print(f"  Impact velocity    : μ={stats.avg_impact_vel:.2f} m/s")
    print(f"  Labels             : {stats.label_summary}")
    print(f"  Rejection rate     : {stats.rejection_rate:.1%}  "
          f"(no_fall={stats.rejected_no_fall}, "
          f"too_short={stats.rejected_too_short})")
    if stats.factor_counts:
        total_active = sum(stats.factor_counts.values())
        factor_line = "  Factor frequency   : " + "  ".join(
            f"{k}={stats.factor_counts[k]/max(total_active,1):.0%}"
            for k in sorted(stats.factor_counts))
        print(factor_line)
    print()


# ---------------------------------------------------------------------------
# Multi‑env parallel collection
# ---------------------------------------------------------------------------

# Per‑environment state bundle.
@dataclass
class _EnvState:
    writer: TrajectoryWriter
    pert: PerturbConfig
    action_queue: list[torch.Tensor] = field(default_factory=list)
    init_h: float = 0.0
    fell: bool = False


_MIN_TRAJ_LEN = 3 * _T2_OFFSET_STEPS + 3  # smallest T where t1=2T/3 < T-5


def _apply_batch_dr(
    env, states, dr_actuator, dr_body,
    robot_cfg, root_body_cfg, device,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Apply PD gain / CoM offset DR to specified envs in one batch call.

    If *env_ids* is None, applies to all envs.  Each distinct (kp, kd)
    pair and each distinct (dx, dy, dz) triplet triggers one batched
    DR call instead of per‑env tiny writes.
    """
    from collections import defaultdict

    if env_ids is None:
        env_ids = torch.arange(len(states), device=device)
    eid_list = env_ids.cpu().tolist()
    if not eid_list:
        return

    kp_groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    com_groups: dict[tuple[float, float, float], list[int]] = defaultdict(list)

    for i in eid_list:
        pert = states[i].pert
        if pert.pd_gain_scale != (1.0, 1.0):
            kp_groups[pert.pd_gain_scale].append(i)
        if pert.com_offset != (0.0, 0.0, 0.0):
            com_groups[pert.com_offset].append(i)

    for (kp_s, kd_s), idxs in kp_groups.items():
        eids = torch.tensor(idxs, device=device)
        dr_actuator.pd_gains(
            env, env_ids=eids,
            kp_range=(kp_s, kp_s), kd_range=(kd_s, kd_s),
            asset_cfg=robot_cfg,
            distribution="uniform", operation="scale",
        )

    for (dx, dy, dz), idxs in com_groups.items():
        eids = torch.tensor(idxs, device=device)
        dr_body.body_com_offset(
            env, env_ids=eids,
            ranges={0: (dx, dx), 1: (dy, dy), 2: (dz, dz)},
            asset_cfg=root_body_cfg,
        )


def collect_trajectories(
    env_cfg,
    output_dir: str | Path,
    num_trajs: int,
    policy_fn: Callable,
    num_envs: int = 1,
    max_steps: int = 500,
    device: str = "cpu",
    seed: int = 42,
    use_terrain: bool = False,
    checkpoint_interval: int = 50,
    use_viewer: bool = False,
) -> None:
    """Run episodes and save falling trajectories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg.scene.num_envs = num_envs
    N = num_envs

    import mujoco
    import mujoco.viewer

    rng = np.random.default_rng(seed)
    env = ManagerBasedRlEnv(env_cfg, device=device)
    asset: Entity = env.scene["robot"]

    # DR helpers.
    from mjlab.envs.mdp.dr import actuator as dr_actuator
    from mjlab.envs.mdp.dr import body as dr_body
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    _robot_cfg = SceneEntityCfg("robot", actuator_names=(".*",))
    _root_body_cfg = SceneEntityCfg("robot", body_names=("torso_link",))

    # ---- Per-env state ----
    states: list[_EnvState] = []
    for _i in range(N):
        pert = _random_perturb(rng)
        writer = TrajectoryWriter(env=env, env_idx=_i)
        states.append(_EnvState(writer=writer, pert=pert))

    # Viewer — renders env 0 via a dedicated MjData synced from Warp.
    # (Multi‑env compositing via mjv_addGeoms cannot render heightfield
    # vertex colours correctly; per‑vertex colour is baked at model‑compile
    # time and not transferable between models.)
    viewer = None
    _v_mjd = None
    if use_viewer:
        _v_mjd = mujoco.MjData(env.sim.mj_model)
        viewer = mujoco.viewer.launch_passive(
            env.sim.mj_model, _v_mjd, key_callback=None)

    def _sync_viewer() -> None:
        if viewer is None:
            return
        sim_data = env.sim.data
        mjm = env.sim.mj_model
        _v_mjd.qpos[:] = sim_data.qpos[0].cpu().numpy()
        _v_mjd.qvel[:] = sim_data.qvel[0].cpu().numpy()
        _v_mjd.xfrc_applied[:] = sim_data.xfrc_applied[0].cpu().numpy()
        mujoco.mj_forward(mjm, _v_mjd)
        viewer.sync()

    # Initial reset — all envs.
    obs_dict, _ = env.reset()
    _apply_batch_dr(env, states, dr_actuator, dr_body,
                _robot_cfg, _root_body_cfg, device)
    frame_batch = extract_predictor_input(env).detach().cpu()  # (N, 63)
    for i in range(N):
        states[i].writer.add_from_batch(frame_batch)
        states[i].init_h = asset.data.root_link_pos_w[i, 2].item()
    del frame_batch
    _sync_viewer()

    stats = CollectStats(start_time=time.time())
    t_start = time.time()

    # ---- Main collection loop ----
    while stats.saved_trajs < num_trajs:
        if viewer is not None and not viewer.is_running():
            print("\n[viewer] closed — stopping collection")
            break

        # --- Action selection (batched) ---
        policy_obs = {k: v.to(device) for k, v in obs_dict.items()
                      if k in ("actor", "critic")}
        # Per‑env sensor noise.
        noise_scale = torch.tensor(
            [s.pert.sensor_noise_scale for s in states],
            device=device).view(N, 1)
        noise_mask = (noise_scale > 1.0).float()
        actor_obs = policy_obs["actor"]
        noise = torch.randn_like(actor_obs) * 0.01 * noise_scale * noise_mask
        policy_obs["actor"] = actor_obs + noise

        actions = policy_fn(policy_obs)  # (N, 29)

        # Per‑env FIFO delay.  Store only CPU copies in the queue;
        # the GPU action tensor is reused in-place each step.
        for i, s in enumerate(states):
            d = s.pert.delay_steps
            if d > 0:
                s.action_queue.append(actions[i].cpu().clone())
                if len(s.action_queue) > d:
                    actions[i] = s.action_queue.pop(0).to(device)
                else:
                    actions[i] = s.action_queue[0].to(device)
        # (no‑delay envs keep the fresh GPU actions as‑is)

        # --- Per‑env perturbations (velocity kicks) — batched ---
        # Collect envs needing velocity changes, then write once.
        perturb_ids: list[int] = []
        perturb_dvx: list[float] = []
        perturb_dvy: list[float] = []

        # External force.
        for i, s in enumerate(states):
            pert = s.pert
            if pert.external_vel != (0.0, 0.0) and len(s.writer) == pert.push_at_step:
                perturb_ids.append(i)
                perturb_dvx.append(pert.external_vel[0])
                perturb_dvy.append(pert.external_vel[1])

        # Foot slip.
        for i, s in enumerate(states):
            pert = s.pert
            if pert.foot_slip_vel != 0.0 and len(s.writer) == pert.slip_at_step:
                direction = float(rng.uniform(0, 2 * np.pi))
                perturb_ids.append(i)
                perturb_dvx.append(pert.foot_slip_vel * np.cos(direction))
                perturb_dvy.append(pert.foot_slip_vel * np.sin(direction))

        if perturb_ids:
            eids = torch.tensor(perturb_ids, device=device)
            rv = asset.data.root_link_vel_w[eids].clone()  # (K, 6)
            rv[:, 0] += torch.tensor(perturb_dvx, device=device)
            rv[:, 1] += torch.tensor(perturb_dvy, device=device)
            asset.write_root_link_velocity_to_sim(rv, env_ids=eids)

        # --- Step all envs ---
        obs_dict, _reward, terminated, truncated, _info = env.step(actions)

        # Extract predictor input ONCE for all envs (not N× per step).
        frame_batch = extract_predictor_input(env).detach().cpu()  # (N, 63)
        for i in range(N):
            states[i].writer.add_from_batch(frame_batch)
        del frame_batch  # free CPU memory promptly
        _sync_viewer()

        # --- Detect falls, max‑steps, reset finished envs ---
        base_h = asset.data.root_link_pos_w[:, 2]  # (N,)
        fell_mask = (base_h < 0.15) & (terminated | truncated).logical_not()
        for i in range(N):
            step_i = len(states[i].writer)
            if fell_mask[i].item() and step_i >= _MIN_TRAJ_LEN:
                states[i].fell = True

        # Emit time_out for envs that hit max_steps.
        env_steps = torch.tensor([len(s.writer) for s in states],
                                 device=device)
        at_max = env_steps >= max_steps

        done_mask = torch.logical_or(
            torch.logical_or(terminated, truncated),
            torch.logical_or(
                torch.tensor([s.fell for s in states], device=device),
                at_max,
            ),
        )

        if done_mask.any().item():
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

            for i in done_ids.cpu().tolist():
                s = states[i]
                traj_len = len(s.writer)
                final_h = base_h[i].item()
                impact_vel = float(
                    torch.norm(asset.data.root_link_lin_vel_w[i]).item())

                stats.total_episodes += 1

                if s.fell and traj_len >= _MIN_TRAJ_LEN:
                    from robot_lab.tasks.safefall.fall_predictor.dataset import compute_labels
                    lbls = compute_labels(traj_len)
                    n_safe = int((lbls == 0).sum().item())
                    n_amb = int((lbls == -1).sum().item())
                    n_fall = int((lbls == 1).sum().item())
                    stats.record_saved(
                        traj_len=traj_len, init_h=s.init_h,
                        final_h=final_h, impact_vel=impact_vel,
                        factors=s.pert.active_factors,
                        n_safe=n_safe, n_amb=n_amb, n_fall=n_fall,
                    )
                    s.writer.save(
                        output_dir / f"traj_{stats.saved_trajs - 1:06d}.pt")
                    factor_str = ", ".join(s.pert.active_factors) if s.pert.active_factors else "none"
                    print(f"\r  [{stats.saved_trajs:5d}] env={i} "
                          f"len={traj_len:3d}  h0={s.init_h:.3f}"
                          f"→h={final_h:.4f}  "
                          f"v_impact={impact_vel:.2f}  "
                          f"safe={n_safe/traj_len:.0%} "
                          f"ambig={n_amb/traj_len:.0%} "
                          f"fall={n_fall/traj_len:.0%}  "
                          f"factors: {factor_str}".ljust(_TERM_WIDTH))
                else:
                    stats.record_rejected(
                        had_fall=s.fell, too_short=(traj_len < _MIN_TRAJ_LEN))

                # New perturbation for this env.
                states[i] = _EnvState(
                    writer=TrajectoryWriter(env=env, env_idx=i),
                    pert=_random_perturb(rng))

            # Apply DR for the newly reset envs (auto_reset already
            _apply_batch_dr(env, states, dr_actuator, dr_body,
                            _robot_cfg, _root_body_cfg, device,
                            env_ids=done_ids)
            frame_batch_r = extract_predictor_input(env).detach().cpu()
            for i in done_ids.cpu().tolist():
                s = states[i]
                s.writer.add_from_batch(frame_batch_r)
                s.init_h = asset.data.root_link_pos_w[i, 2].item()
            del frame_batch_r
            _sync_viewer()

        # --- Checkpoints & progress ---
        if stats.saved_trajs > 0 and stats.saved_trajs % checkpoint_interval == 0:
            _print_checkpoint(stats, num_trajs, t_start)
        if stats.saved_trajs > 0:
            _print_progress(stats, num_trajs, t_start)

    if viewer is not None:
        viewer.close()
    env.close()
    elapsed = time.time() - t_start

    # Final summary.
    print(f"\n{_HR}\n  COLLECTION COMPLETE\n{_HR}")
    print(f"  Total episodes      : {stats.total_episodes}")
    print(f"  Trajectories saved  : {stats.saved_trajs}")
    print(f"  Rejected (no fall)  : {stats.rejected_no_fall}")
    print(f"  Rejected (short)    : {stats.rejected_too_short}")
    print(f"  Acceptance rate     : {stats.saved_trajs/max(stats.total_episodes,1):.1%}")
    print(f"  Wall time           : {_fmt_duration(elapsed)}")
    print(f"  Avg rate            : {stats.saved_trajs/max(elapsed,1):.2f} traj/s")
    lengths = stats.traj_lengths
    if lengths:
        print(f"  Trajectory lengths  : μ={stats.avg_traj_len:.0f}  "
              f"[{min(lengths)}, {max(lengths)}]")
    print(f"  Init height (m)     : μ={stats.avg_init_height:.3f}")
    print(f"  Final height (m)    : μ={stats.avg_final_height:.4f}")
    print(f"  Impact velocity     : μ={stats.avg_impact_vel:.2f} m/s")
    print(f"  Labels (avg)        : {stats.label_summary}")
    if stats.factor_counts:
        total_active = sum(stats.factor_counts.values())
        print(f"  Factor frequency    : " + "  ".join(
            f"{k}={stats.factor_counts[k]/max(total_active,1):.1%}"
            for k in sorted(stats.factor_counts)))
    print(_HR)
    _save_report(output_dir, stats, num_trajs, elapsed, use_terrain)


def _save_report(
    output_dir: Path, stats: CollectStats, target: int,
    elapsed: float, use_terrain: bool,
) -> None:
    report_path = output_dir / "REPORT.txt"
    lines: list[str] = []

    def L(s: str = "") -> None:
        lines.append(s)

    L("=" * 72)
    L("  SafeFall Trajectory Collection Report")
    L("=" * 72)
    L()
    L(f"  Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"  Output dir   : {output_dir.resolve()}")
    L(f"  Target count : {target}")
    L("  Policy       : nominal velocity policy")
    L(f"  Terrain      : {'rough (generator)' if use_terrain else 'flat plane'}")
    L()
    L("─" * 72)
    L("  Collection Summary")
    L("─" * 72)
    L(f"  Total episodes          : {stats.total_episodes}")
    L(f"  Trajectories saved      : {stats.saved_trajs}")
    L(f"  Rejected (no fall)      : {stats.rejected_no_fall}")
    L(f"  Rejected (too short)    : {stats.rejected_too_short}")
    L(f"  Acceptance rate         : {stats.saved_trajs/max(stats.total_episodes,1):.1%}")
    L(f"  Wall time               : {_fmt_duration(elapsed)}")
    L(f"  Avg collection rate     : {stats.saved_trajs/max(elapsed,1):.2f} traj/s")
    L()
    L("─" * 72)
    L("  Trajectory Statistics")
    L("─" * 72)
    if stats.traj_lengths:
        lengths = np.array(stats.traj_lengths)
        L(f"  Length (steps)  : μ={lengths.mean():.1f}  σ={lengths.std():.1f}  "
          f"min={lengths.min()}  max={lengths.max()}  "
          f"p25={np.percentile(lengths,25):.0f}  p50={np.percentile(lengths,50):.0f}  "
          f"p75={np.percentile(lengths,75):.0f}")
        L(f"  Duration (s)    : μ={lengths.mean()*0.02:.2f}  "
          f"min={lengths.min()*0.02:.2f}  max={lengths.max()*0.02:.2f}")
        heights = np.array(stats.init_heights)
        L(f"  Init height (m) : μ={heights.mean():.3f}  σ={heights.std():.3f}  "
          f"min={heights.min():.3f}  max={heights.max():.3f}")
        final_h = np.array(stats.final_heights)
        L(f"  Final height (m): μ={final_h.mean():.4f}  σ={final_h.std():.4f}  "
          f"min={final_h.min():.4f}  max={final_h.max():.4f}")
        imp = np.array(stats.impact_velocities)
        L(f"  Impact vel (m/s): μ={imp.mean():.2f}  σ={imp.std():.2f}  "
          f"min={imp.min():.2f}  max={imp.max():.2f}")
    else:
        L("  (no trajectories saved)")
    L()
    L("─" * 72)
    L("  Label Distribution (averaged over trajectories)")
    L("─" * 72)
    L(f"  {stats.label_summary}")
    L(f"  Total label frames : {stats.label_safe + stats.label_ambiguous + stats.label_falling}")
    L()
    L("─" * 72)
    L("  Perturbation Factor Frequency")
    L("─" * 72)
    total_active = sum(stats.factor_counts.values())
    if total_active > 0:
        for k in sorted(stats.factor_counts):
            pct = stats.factor_counts[k] / total_active * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            L(f"  {k:<22s} {bar} {stats.factor_counts[k]:5d} ({pct:5.1f}%)")
    else:
        L("  (no perturbation factors recorded)")
    L(f"  Total activations   : {total_active}")
    L(f"  Avg factors / traj  : {total_active/max(stats.saved_trajs,1):.2f}")
    L()
    L("─" * 72)
    L("  File Inventory")
    L("─" * 72)
    pt_files = sorted(output_dir.glob("traj_*.pt"))
    L(f"  .pt trajectory files : {len(pt_files)}")
    if pt_files:
        for f in pt_files[:3]:
            L(f"    {f.name}")
        if len(pt_files) > 6:
            L(f"    ... ({len(pt_files) - 6} files)")
        for f in pt_files[-3:]:
            L(f"    {f.name}")
    L()
    L("=" * 72)
    L("  End of Report")
    L("=" * 72)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# Custom rough terrain (paper foot‑trip spec)
# ---------------------------------------------------------------------------

def _build_foot_trip_terrain(base):
    from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
    from mjlab.terrains.heightfield_terrains import (
        HfDiscreteObstaclesTerrainCfg,
        HfRandomUniformTerrainCfg,
        HfWaveTerrainCfg,
    )
    _CELL, _HMAX = (8.0, 8.0), 0.15
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=None, curriculum=False, size=_CELL,
            num_rows=5, num_cols=5, border_width=10.0,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.4, size=_CELL,
                    noise_range=(0.02, _HMAX), noise_step=0.005,
                    horizontal_scale=0.1, vertical_scale=0.005,
                ),
                "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
                    proportion=0.3, size=_CELL,
                    obstacle_height_range=(0.03, _HMAX),
                    obstacle_width_range=(0.2, 1.0),
                    num_obstacles=15, platform_width=0.5,
                    horizontal_scale=0.1, vertical_scale=0.005,
                ),
                "waves": HfWaveTerrainCfg(
                    proportion=0.3, size=_CELL,
                    amplitude_range=(0.03, _HMAX), num_waves=3,
                    horizontal_scale=0.1, vertical_scale=0.005,
                ),
            },
        ),
        env_spacing=base.env_spacing,
        num_envs=base.num_envs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect falling trajectories")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-trajs", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-checkpoint", type=str, required=True)
    parser.add_argument("--policy-task", type=str,
                        default="Mjlab-Velocity-Flat-Unitree-G1")
    parser.add_argument("--rough", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    # Build env config.
    if args.rough:
        env_cfg = load_env_cfg(args.policy_task, play=True)
        env_cfg.scene.terrain = _build_foot_trip_terrain(env_cfg.scene.terrain)
        use_terrain = True
    else:
        env_cfg = load_env_cfg(args.policy_task, play=True)
        use_terrain = False

    # Load nominal policy.
    from dataclasses import asdict
    rl_cfg = load_rl_cfg(args.policy_task)
    runner_cls = load_runner_cls(args.policy_task) or MjlabOnPolicyRunner

    policy_env_cfg = load_env_cfg(args.policy_task, play=True)
    policy_env_cfg.scene.num_envs = 1

    temp_env = ManagerBasedRlEnv(policy_env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(temp_env)
    runner = runner_cls(wrapped, asdict(rl_cfg), log_dir="/tmp",
                        device=args.device)
    runner.load(args.policy_checkpoint, load_cfg={"actor": True},
                strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    def policy_fn(obs_dict):
        return policy(obs_dict)

    temp_env.close()

    _print_header({
        "policy_task": args.policy_task,
        "policy_ckpt": args.policy_checkpoint,
        "terrain": "rough" if use_terrain else "flat",
        "target": args.num_trajs,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "device": args.device,
        "seed": args.seed,
    }, output_dir=args.output)

    collect_trajectories(
        env_cfg=env_cfg,
        output_dir=args.output,
        num_trajs=args.num_trajs,
        policy_fn=policy_fn,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        use_terrain=use_terrain,
        checkpoint_interval=args.checkpoint_interval,
        use_viewer=args.viewer,
    )


if __name__ == "__main__":
    main()
