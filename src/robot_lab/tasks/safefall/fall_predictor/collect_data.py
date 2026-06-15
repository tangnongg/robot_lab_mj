#!/usr/bin/env python3
"""Collect falling trajectories for fall predictor training — paper §III-B.

Runs a nominal locomotion policy under perturbation to induce falls,
recording proprioceptive observations up to ground impact.

Real-time progress displays per-trajectory detail plus periodic summaries.
After collection, a REPORT.txt is written to the output directory.

Paper Table I — 6 failure factors implemented:
  1. Sensor noise      — scale observation noise 2-10×
  2. External force     — velocity perturbation to torso
  3. Foot slip          — velocity kick to stance foot body
  4. Foot trip          — rough terrain (--rough flag)
  5. System delay       — hold last action for [0, 200] ms
  6. Dynamic mismatch   — PD gain randomization + CoM offset at reset

Usage:
    # Flat terrain
    python -m robot_lab.tasks.safefall.fall_predictor.collect_data \\
        --output data/fall_trajs --num-trajs 5000 \\
        --policy-checkpoint logs/rsl_rl/.../model_1999.pt \\
        --policy-task Mjlab-Velocity-Flat-Unitree-G1

    # Rough terrain (foot trip factor)
    python -m robot_lab.tasks.safefall.fall_predictor.collect_data \\
        --output data/fall_trajs_rough --num-trajs 5000 \\
        --policy-checkpoint logs/rsl_rl/.../model_1999.pt \\
        --policy-task Mjlab-Velocity-Flat-Unitree-G1 --rough

    # With viewer
    python -m robot_lab.tasks.safefall.fall_predictor.collect_data \\
        --output data/fall_trajs --num-trajs 500 \\
        --policy-checkpoint logs/rsl_rl/.../model_1999.pt --viewer
"""

from __future__ import annotations

import argparse
import json
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
from robot_lab.tasks.safefall.fall_predictor.model import INPUT_DIM

# Suppress mjlab INFO logging during collection (tables flood terminal).
import logging
logging.getLogger("mjlab").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

# Human-readable factor names.
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
    """Configuration for a single perturbation episode.

    Fields ordered per paper Table I / FACTOR_NAMES:
      0: sensor_noise
      1: external_force
      2: foot_slip
      3: foot_trip  — handled by ``use_terrain`` flag, no field here
      4: system_delay
      5: dynamic_mismatch
    """

    # 0 — sensor noise
    sensor_noise_scale: float = 1.0

    # 1 — external force (velocity perturbation to torso)
    external_vel: tuple[float, float] = (0.0, 0.0)  # (vx, vy) m/s
    push_at_step: int = 50

    # 2 — foot slip
    foot_slip_vel: float = 0.0
    slip_at_step: int = 60

    # 3 — foot trip (see ``use_terrain`` flag)

    # 4 — system delay
    delay_steps: int = 0

    # 5 — dynamic mismatch
    pd_gain_scale: tuple[float, float] = (1.0, 1.0)
    com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def active_factors(self) -> list[str]:
        """Return list of active factor names, in FACTOR_NAMES order."""
        names: list[str] = []

        # 0 — sensor_noise
        if self.sensor_noise_scale > 1.0:
            names.append(f"sensor_noise({self.sensor_noise_scale:.1f}×)")

        # 1 — external_force
        if self.external_vel != (0.0, 0.0):
            names.append(
                f"external_force(vx={self.external_vel[0]:.1f},"
                f"vy={self.external_vel[1]:.1f}m/s)"
            )

        # 2 — foot_slip
        if self.foot_slip_vel != 0.0:
            names.append(f"foot_slip({self.foot_slip_vel:.1f}m/s)")

        # 3 — foot_trip (use_terrain)

        # 4 — system_delay
        if self.delay_steps > 0:
            names.append(f"delay({self.delay_steps * 20}ms)")

        # 5 — dynamic_mismatch
        if self.pd_gain_scale != (1.0, 1.0) or self.com_offset != (0.0, 0.0, 0.0):
            parts: list[str] = []
            if self.pd_gain_scale != (1.0, 1.0):
                parts.append(
                    f"kp{self.pd_gain_scale[0]:.1f}/kd{self.pd_gain_scale[1]:.1f}"
                )
            if self.com_offset != (0.0, 0.0, 0.0):
                parts.append(
                    f"CoM({self.com_offset[0]:.2f},{self.com_offset[1]:.2f})"
                )
            names.append("dyn_mismatch(" + ",".join(parts) + ")")

        return names


def _random_perturb(rng: np.random.Generator) -> PerturbConfig:
    """Sample a random perturbation configuration (1-3 factors).

    Factor 3 (foot_trip) is never sampled — it is controlled by the
    ``--rough`` flag at the env level (rough terrain is always active
    when requested, not randomized per episode).
    """
    cfg = PerturbConfig()
    available = [0, 1, 2, 4, 5]
    n_factors = rng.integers(1, min(4, len(available) + 1))
    factors = rng.choice(available, size=n_factors, replace=False)

    for f in factors:
        if f == 0:  # sensor noise
            cfg.sensor_noise_scale = float(rng.uniform(2.0, 10.0))
        elif f == 1:  # external force (velocity perturbation to torso)
            cfg.external_vel = (
                float(rng.uniform(-2.0, 2.0)),  # x: forward/backward
                float(rng.uniform(-1.0, 1.0)),  # y: lateral
            )
            cfg.push_at_step = int(rng.integers(20, 80))
        elif f == 2:  # foot slip
            cfg.foot_slip_vel = float(rng.uniform(-1.5, 1.5))
            cfg.slip_at_step = int(rng.integers(20, 80))
        elif f == 4:  # system delay
            cfg.delay_steps = int(rng.integers(1, 11))
        elif f == 5:  # dynamic mismatch
            # Paper: stiffness logU(0.7, 1.5), damping logU(0.5, 3.0).
            # log-uniform = exp(uniform(log(lo), log(hi))) — skews toward 1.0×.
            cfg.pd_gain_scale = (
                float(np.exp(rng.uniform(np.log(0.7), np.log(1.5)))),
                float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))),
            )
            cfg.com_offset = (
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.01, 0.01),
            )
    return cfg


# ---------------------------------------------------------------------------
# Statistics accumulator
# ---------------------------------------------------------------------------

@dataclass
class CollectStats:
    """Running statistics across all episodes."""
    # Counters
    total_episodes: int = 0
    saved_trajs: int = 0
    rejected_no_fall: int = 0
    rejected_too_short: int = 0

    # Trajectory lengths (saved only)
    traj_lengths: list[int] = field(default_factory=list)
    # Initial base heights (saved only)
    init_heights: list[float] = field(default_factory=list)
    # Final base heights
    final_heights: list[float] = field(default_factory=list)
    # Impact velocities (magnitude of root lin vel at last step)
    impact_velocities: list[float] = field(default_factory=list)

    # Factor activation counts
    factor_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Timing
    start_time: float = 0.0

    # Labels distribution (accumulate over all saved trajs)
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
_THIN = "─" * _TERM_WIDTH


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_header(args_dict: dict, output_dir: str) -> None:
    print()
    print(_HR)
    print("  SafeFall Data Collection  |  paper §III-B")
    print(_HR)
    for k, v in args_dict.items():
        print(f"  {k:<22s}: {v}")
    print(f"  {'output':<22s}: {output_dir}")
    print(_HR)
    print()


def _print_progress(stats: CollectStats, target: int, t_start: float):
    """Print one-line progress bar + key stats."""
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
    # Pad to terminal width to overwrite previous line.
    print(line.ljust(_TERM_WIDTH), end="", flush=True)


def _print_checkpoint(stats: CollectStats, target: int, t_start: float) -> None:
    """Print a detailed checkpoint block every N trajectories."""
    elapsed = time.time() - t_start
    print()
    print(f"  ── Checkpoint @ {stats.saved_trajs}/{target} "
          f"({_fmt_duration(elapsed)} elapsed) " + "─" * 40)
    print(f"  Trajectory lengths : μ={stats.avg_traj_len:.0f}  "
          f"min={min(stats.traj_lengths) if stats.traj_lengths else 0}  "
          f"max={max(stats.traj_lengths) if stats.traj_lengths else 0}")
    print(f"  Init height (m)    : μ={stats.avg_init_height:.3f}")
    print(f"  Final height (m)   : μ={stats.avg_final_height:.4f}")
    print(f"  Impact velocity    : μ={stats.avg_impact_vel:.2f} m/s")
    print(f"  Labels             : {stats.label_summary}")
    print(f"  Rejection rate     : {stats.rejection_rate:.1%}  "
          f"(no_fall={stats.rejected_no_fall}, too_short={stats.rejected_too_short})")

    # Factor frequency
    if stats.factor_counts:
        total_active = sum(stats.factor_counts.values())
        factor_line = "  Factor frequency   : " + "  ".join(
            f"{k}={stats.factor_counts[k]/max(total_active,1):.0%}"
            for k in sorted(stats.factor_counts)
        )
        print(factor_line)
    print()


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_trajectories(
    env_cfg,
    output_dir: str | Path,
    num_trajs: int,
    policy_fn: Callable,
    max_steps: int = 500,
    device: str = "cpu",
    seed: int = 42,
    use_terrain: bool = False,
    checkpoint_interval: int = 50,
    use_viewer: bool = False,
) -> None:
    """Run episodes and save falling trajectories.

    Parameters
    ----------
    checkpoint_interval : int
        Print detailed checkpoint every N saved trajectories.
    use_viewer : bool
        If True, open a MuJoCo viewer window.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import mujoco
    import mujoco.viewer

    rng = np.random.default_rng(seed)
    env = ManagerBasedRlEnv(env_cfg, device=device)
    asset: Entity = env.scene["robot"]

    # Viewer — sync Warp→mjData before each render frame.
    # mjData is NOT auto-updated by the Warp backend; we must manually
    # copy qpos/qvel/xfrc_applied from Warp tensors before mj_forward+sync.
    viewer = None
    mjm = mjd = None  # type: ignore
    if use_viewer:
        mjm = env.sim.mj_model
        mjd = env.sim.mj_data
        viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=None)

    def _sync_viewer() -> None:
        """Copy Warp tensors → mjData → mj_forward → viewer sync."""
        if viewer is None:
            return
        sim_data = env.sim.data
        mjd.qpos[:] = sim_data.qpos[0].cpu().numpy()
        mjd.qvel[:] = sim_data.qvel[0].cpu().numpy()
        mjd.xfrc_applied[:] = sim_data.xfrc_applied[0].cpu().numpy()
        mujoco.mj_forward(mjm, mjd)
        viewer.sync()

    # DR helpers.
    from mjlab.envs.mdp.dr import actuator as dr_actuator
    from mjlab.envs.mdp.dr import body as dr_body
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    _robot_cfg = SceneEntityCfg("robot", actuator_names=(".*",))
    _root_body_cfg = SceneEntityCfg("robot", body_names=("torso_link",))

    stats = CollectStats(start_time=time.time())
    t_start = time.time()

    while stats.saved_trajs < num_trajs:
        if viewer is not None and not viewer.is_running():
            print("\n[viewer] closed — stopping collection")
            break

        stats.total_episodes += 1

        pert = _random_perturb(rng)

        # Reset + apply dynamic mismatch.
        obs_dict, _ = env.reset()
        _sync_viewer()

        if pert.pd_gain_scale != (1.0, 1.0):
            kp_s, kd_s = pert.pd_gain_scale
            dr_actuator.pd_gains(
                env, env_ids=None,
                kp_range=(kp_s, kp_s), kd_range=(kd_s, kd_s),
                asset_cfg=_robot_cfg,
                distribution="uniform", operation="scale",
            )
        if pert.com_offset != (0.0, 0.0, 0.0):
            dx, dy, dz = pert.com_offset
            # body_com_offset: integer-keyed ranges → per-axis control.
            dr_body.body_com_offset(
                env, env_ids=None,
                ranges={0: (dx, dx), 1: (dy, dy), 2: (dz, dz)},
                asset_cfg=_root_body_cfg,
            )

        writer = TrajectoryWriter(env=env)
        writer.add()

        # Initial height.
        init_h = asset.data.root_link_pos_w[0, 2].item()

        # Action delay: FIFO pipeline.  Every step policy runs with
        # fresh observation, but the action is queued and pops out
        # delay_steps later — matching the paper's "delay in the
        # observation-to-action loop".
        action_queue: list[torch.Tensor] = []

        # Per-episode step counter for perturbation timing.
        fell = False
        step = 0

        for step in range(max_steps):
            # — action selection —
            policy_obs = {k: v.to(device) for k, v in obs_dict.items()
                          if k in ("actor", "critic")}
            if pert.sensor_noise_scale > 1.0:
                noise = torch.randn_like(policy_obs["actor"]) * 0.01 * pert.sensor_noise_scale
                policy_obs["actor"] = policy_obs["actor"] + noise

            if pert.delay_steps > 0:
                # policy runs every step — sees fresh observation.
                action_queue.append(policy_fn(policy_obs))
                # Actuator receives the action that entered the queue
                # delay_steps ago (FIFO pipeline delay).
                if len(action_queue) > pert.delay_steps:
                    action = action_queue.pop(0)
                else:
                    action = action_queue[0]  # warmup: queue not full yet
            else:
                action = policy_fn(policy_obs)

            # — external force (velocity perturbation to torso) —
            if pert.external_vel != (0.0, 0.0) and step == pert.push_at_step:
                vx, vy = pert.external_vel
                root_vel = asset.data.root_link_vel_w[0:1].clone()
                root_vel[0, 0] += vx
                root_vel[0, 1] += vy
                asset.write_root_link_velocity_to_sim(
                    root_vel, env_ids=torch.tensor([0])
                )

            # — foot slip —
            if pert.foot_slip_vel != 0.0 and step == pert.slip_at_step:
                direction = float(rng.uniform(0, 2 * np.pi))
                vx = pert.foot_slip_vel * np.cos(direction)
                vy = pert.foot_slip_vel * np.sin(direction)
                root_vel = asset.data.root_link_vel_w[0:1].clone()
                root_vel[0, 0] += vx
                root_vel[0, 1] += vy
                asset.write_root_link_velocity_to_sim(root_vel, env_ids=torch.tensor([0]))

            # Step env.
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            writer.add()

            # Sync viewer (Warp→mjData→mj_forward→render).
            _sync_viewer()

            # Episode termination
            base_h = asset.data.root_link_pos_w[0, 2].item()
            if base_h < 0.15 and step >= 3 * _T2_OFFSET_STEPS + 2:
                fell = True  # fall_completed
                break
            
            if terminated[0] or truncated[0]:
                break

        # Episode outcome.
        traj_len = len(writer)
        final_h = asset.data.root_link_pos_w[0, 2].item()
        impact_vel = float(torch.norm(asset.data.root_link_lin_vel_w[0]).item())

        if fell and traj_len >= 10:
            # Compute label distribution from the on-disk tensor.
            labels = writer.tensor  # just for length; labels are computed at save time
            from robot_lab.tasks.safefall.fall_predictor.dataset import compute_labels
            lbls = compute_labels(traj_len)
            n_safe = int((lbls == 0).sum().item())
            n_amb = int((lbls == -1).sum().item())
            n_fall = int((lbls == 1).sum().item())

            stats.record_saved(
                traj_len=traj_len,
                init_h=init_h,
                final_h=final_h,
                impact_vel=impact_vel,
                factors=pert.active_factors,
                n_safe=n_safe,
                n_amb=n_amb,
                n_fall=n_fall,
            )

            writer.save(output_dir / f"traj_{stats.saved_trajs - 1:06d}.pt")

            # Per-trajectory line.
            factor_str = ", ".join(pert.active_factors) if pert.active_factors else "none"
            print(f"\r  [{stats.saved_trajs:5d}] len={traj_len:3d}  "
                  f"h0={init_h:.3f}→h={final_h:.4f}  "
                  f"v_impact={impact_vel:.2f}  "
                  f"safe={n_safe/traj_len:.0%} ambig={n_amb/traj_len:.0%} fall={n_fall/traj_len:.0%}  "
                  f"factors: {factor_str}".ljust(_TERM_WIDTH))
        else:
            reason = "no_fall" if not fell else "too_short"
            stats.record_rejected(had_fall=fell, too_short=(traj_len < 10))
            # Short rejection line only at higher verbosity (skip clutter).
            if traj_len < 10:
                pass  # silently skip very short rejects
            elif stats.total_episodes % 10 == 0:
                print(f"\r  [rej] episode {stats.total_episodes}: {reason} "
                      f"(len={traj_len}, final_h={final_h:.3f})".ljust(_TERM_WIDTH))

        # Detailed checkpoint every N saved trajectories.
        if stats.saved_trajs > 0 and stats.saved_trajs % checkpoint_interval == 0:
            _print_checkpoint(stats, num_trajs, t_start)

        # Live progress bar (overwrites previous line).
        if stats.saved_trajs > 0:
            _print_progress(stats, num_trajs, t_start)

    if viewer is not None:
        viewer.close()
    env.close()
    elapsed = time.time() - t_start

    # Final summary.
    print()
    print(_HR)
    print(f"  COLLECTION COMPLETE")
    print(_HR)
    print(f"  Total episodes      : {stats.total_episodes}")
    print(f"  Trajectories saved  : {stats.saved_trajs}")
    print(f"  Rejected (no fall)  : {stats.rejected_no_fall}")
    print(f"  Rejected (short)    : {stats.rejected_too_short}")
    print(f"  Acceptance rate     : {stats.saved_trajs/max(stats.total_episodes,1):.1%}")
    print(f"  Wall time           : {_fmt_duration(elapsed)}")
    print(f"  Avg rate            : {stats.saved_trajs/max(elapsed,1):.2f} traj/s")
    print(f"  Trajectory lengths  : μ={stats.avg_traj_len:.0f}  "
          f"[{min(stats.traj_lengths) if stats.traj_lengths else 0}, "
          f"{max(stats.traj_lengths) if stats.traj_lengths else 0}]")
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

    # Save REPORT.txt.
    _save_report(output_dir, stats, num_trajs, elapsed, use_terrain)


def _save_report(
    output_dir: Path,
    stats: CollectStats,
    target: int,
    elapsed: float,
    use_terrain: bool,
) -> None:
    """Write a comprehensive plain-text report to ``REPORT.txt``."""
    report_path = output_dir / "REPORT.txt"

    lines = []

    def L(s=""):
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
    L("  Seed         : (see CLI)")
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
    L(f"  (t ≤ 2T/3 → safe, 2T/3 < t ≤ T−100ms → ambiguous, t > T−100ms → falling)")

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
        # Show first 3 and last 3.
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
# CLI
# ---------------------------------------------------------------------------

def _build_foot_trip_terrain(base: TerrainEntityCfg) -> TerrainEntityCfg:
    """Build a terrain config matching the paper's foot-trip specification.

    Paper Table I: "unseen terrain heightfields and obstacles with heights
    ranging from [0, 15] cm."

    Uses heightfield-only sub-terrains (no box primitives, no flat), with
    all obstacle heights strictly within [0, 0.15] m.
    """
    from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
    from mjlab.terrains.heightfield_terrains import (
        HfDiscreteObstaclesTerrainCfg,
        HfRandomUniformTerrainCfg,
        HfWaveTerrainCfg,
    )

    _CELL = (8.0, 8.0)
    _HMAX = 0.15  # paper: [0, 15] cm

    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=None,
            curriculum=False,
            size=_CELL,
            num_rows=5,
            num_cols=5,
            border_width=10.0,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                # Random rough bumps, 2–15 cm — core heightfield perturbation.
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    size=_CELL,
                    noise_range=(0.02, _HMAX),
                    noise_step=0.005,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                ),
                # Discrete obstacles (blocks / curbs), 3–15 cm.
                "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
                    proportion=0.3,
                    size=_CELL,
                    obstacle_height_range=(0.03, _HMAX),
                    obstacle_width_range=(0.2, 1.0),
                    num_obstacles=15,
                    platform_width=0.5,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                ),
                # Sinusoidal waves, amplitude 3–15 cm.
                "waves": HfWaveTerrainCfg(
                    proportion=0.3,
                    size=_CELL,
                    amplitude_range=(0.03, _HMAX),
                    num_waves=3,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                ),
            },
        ),
        env_spacing=base.env_spacing,
        num_envs=base.num_envs,
    )


def main():
    parser = argparse.ArgumentParser(description="Collect falling trajectories")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num-trajs", type=int, default=1000, help="Number of trajectories")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-checkpoint", type=str, required=True,
                        help="Path to nominal policy checkpoint")
    parser.add_argument("--policy-task", type=str,
                        default="Mjlab-Velocity-Flat-Unitree-G1",
                        help="Task ID for nominal policy env")
    parser.add_argument("--rough", action="store_true",
                        help="Use rough terrain (foot trip factor, paper Table I)")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Print detailed checkpoint every N saved trajectories")
    parser.add_argument("--viewer", action="store_true",
                        help="Open MuJoCo viewer during collection")
    args = parser.parse_args()

    # Build env config.
    if args.rough:
        env_cfg = load_env_cfg(args.policy_task, play=True)
        env_cfg.scene.terrain = _build_foot_trip_terrain(env_cfg.scene.terrain)
        use_terrain = True
    else:
        env_cfg = load_env_cfg(args.policy_task, play=True)
        use_terrain = False
    env_cfg.scene.num_envs = 1

    # Load nominal policy.
    from dataclasses import asdict
    rl_cfg = load_rl_cfg(args.policy_task)
    runner_cls = load_runner_cls(args.policy_task) or MjlabOnPolicyRunner

    policy_env_cfg = load_env_cfg(args.policy_task, play=True)
    policy_env_cfg.scene.num_envs = 1

    temp_env = ManagerBasedRlEnv(policy_env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(temp_env)
    runner = runner_cls(wrapped, asdict(rl_cfg), log_dir="/tmp", device=args.device)
    runner.load(args.policy_checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    def policy_fn(obs_dict):
        return policy(obs_dict)

    temp_env.close()

    # Print header with config summary.
    _print_header(
        {
            "policy_task": args.policy_task,
            "policy_ckpt": args.policy_checkpoint,
            "terrain": "rough" if use_terrain else "flat",
            "target": args.num_trajs,
            "max_steps": args.max_steps,
            "device": args.device,
            "seed": args.seed,
        },
        output_dir=args.output,
    )

    collect_trajectories(
        env_cfg=env_cfg,
        output_dir=args.output,
        num_trajs=args.num_trajs,
        policy_fn=policy_fn,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        use_terrain=use_terrain,
        checkpoint_interval=args.checkpoint_interval,
        use_viewer=args.viewer,
    )


if __name__ == "__main__":
    main()
