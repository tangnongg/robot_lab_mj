#!/usr/bin/env python3
"""Prepare Stage II initial states for SafeFall policy training.

Paper §III-D: "In Stage II, we sample states from our collected dataset
that the fall predictor classifies as unsafe."

This script:
  1. Loads all falling trajectories from --traj-dir
  2. Runs the fall predictor over each trajectory to find frames classified
     as "falling" (is_falling == True)
  3. For each flagged frame, extracts the full simulation state
     (root position/orientation/velocity + joint positions/velocities)
  4. Filters states that are already too close to impact
  5. Saves a state bank to --output for use by the Stage II reset event

Usage:
    python -m robot_lab.tasks.safefall.fall_predictor.prepare_stage2_states \
        --traj-dir data/fall_trajs \
        --predictor models/fall_predictor.pt \
        --output models/stage2_state_bank.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_src = Path(__file__).resolve().parents[4]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from robot_lab.tasks.safefall.fall_predictor.dataset import load_trajectory
from robot_lab.tasks.safefall.fall_predictor.model import FallPredictor


def extract_stage2_states(
    traj_dir: str | Path,
    predictor_ckpt: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
    confirmation_steps: int = 3,
    threshold: float = 0.5,
    warmup_steps: int = 10,
    min_height: float = 0.3,
    max_height: float = 1.2,
) -> dict[str, torch.Tensor]:
    """Extract predictor-flagged falling states from trajectories.

    Returns
    -------
    dict with keys:
      root_state  — (K, 13)  [x,y,z, qw,qx,qy,qz, vx,vy,vz, ωx,ωy,ωz]
      joint_pos   — (K, 29)  absolute joint positions
      joint_vel   — (K, 29)  joint velocities
    """
    traj_dir = Path(traj_dir)
    predictor_ckpt = Path(predictor_ckpt)
    output_path = Path(output_path)

    files = sorted(traj_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt trajectory files found in {traj_dir}")
    print(f"Found {len(files)} trajectories")

    # Load predictor
    model = FallPredictor()
    model.load(predictor_ckpt)
    model.to(device)
    model.eval()
    print("Predictor loaded")

    all_root: list[torch.Tensor] = []
    all_jpos: list[torch.Tensor] = []
    all_jvel: list[torch.Tensor] = []

    total_flagged = 0
    total_frames = 0

    for fi, fpath in enumerate(files):
        d = load_trajectory(fpath)
        required_state = {"root_state", "joint_pos", "joint_vel"}
        if not required_state.issubset(d) or d.get("state_schema_version", 0) < 2:
            raise ValueError(
                f"{fpath} has predictor observations but no exact simulator state. "
                "Root height, linear velocity, and yaw cannot be reconstructed from "
                "the 63-D predictor input. Recollect trajectories with the current "
                "collector before building a Stage II bank."
            )
        obs_seq = d["observations"]  # (T, 63)
        T = obs_seq.shape[0]
        total_frames += T

        # --- Run predictor sequentially over the trajectory ---
        hidden = None
        consec = 0
        is_falling = False

        for t in range(T):
            x = obs_seq[t:t + 1].to(device)  # (1, 63)
            with torch.no_grad():
                prob, hidden = model.predict_proba(x, hidden)

            p = float(prob.item())
            if t < warmup_steps:
                continue

            if p > threshold:
                consec += 1
            else:
                consec = 0

            if consec >= confirmation_steps:
                is_falling = True

            if not is_falling:
                continue

            root_state = d["root_state"][t].clone()
            jpos_abs = d["joint_pos"][t].clone()
            jvel = d["joint_vel"][t].clone()
            if not (min_height <= float(root_state[2]) <= max_height):
                continue

            # Simple validity filter: reject extreme joint positions.
            if jpos_abs.abs().max() > 5.0:
                continue

            all_root.append(root_state)
            all_jpos.append(jpos_abs)
            all_jvel.append(jvel)
            total_flagged += 1

        if (fi + 1) % 500 == 0:
            print(f"  {fi+1}/{len(files)} trajs, {total_flagged} flagged so far")

    if not all_root:
        raise RuntimeError("No falling states extracted — check predictor/threshold.")

    root_state_bank = torch.stack(all_root)       # (K, 13)
    joint_pos_bank = torch.stack(all_jpos)         # (K, 29)
    joint_vel_bank = torch.stack(all_jvel)         # (K, 29)

    print(f"\nTotal frames processed: {total_frames}")
    print(f"Stage II state bank: {len(all_root)} states")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "root_state": root_state_bank,
        "joint_pos": joint_pos_bank,
        "joint_vel": joint_vel_bank,
        "K": len(all_root),
        "state_schema_version": 2,
    }
    torch.save(data, output_path)
    print(f"Saved to {output_path}")
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Stage II initial states for SafeFall")
    parser.add_argument("--traj-dir", type=str, required=True)
    parser.add_argument("--predictor", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/stage2_state_bank.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--confirmation-steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    extract_stage2_states(
        traj_dir=args.traj_dir,
        predictor_ckpt=args.predictor,
        output_path=args.output,
        device=args.device,
        threshold=args.threshold,
        confirmation_steps=args.confirmation_steps,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
