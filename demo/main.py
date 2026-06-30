#!/usr/bin/env python3
"""SafeFall interactive demo: joystick‑controlled walking + push + stand‑up.

Default mode: G1 walks forward under velocity control.
             A fall predictor monitors proprioception continuously.

- Press **A**: inject a random push (may cause a fall).
- Press **B**: trigger stand‑up (HoST policy).
- When predictor detects a fall → auto‑switches to SafeFall policy.

Usage
-----
    python -m demo.main             # uses CPU
    python -m demo.main --device cuda:0   # GPU (faster)
"""

from __future__ import annotations

import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj / "src") not in sys.path:
    sys.path.insert(0, str(_proj / "src"))

from mjlab.tasks.registry import load_env_cfg

from demo.policies import (
    load_velocity_policy,
    load_safefall_policy,
    load_host_policy,
    load_predictor_wrapper,
)
from demo.runner import run_interactive

_VELOCITY_TASK = "Mjlab-Velocity-Flat-Unitree-G1"
_SAFEFALL_TASK = "Mjlab-SafeFall-G1"
_HOST_TASK = "Mjlab-HoST-Ground-Unitree-G1"


def main(device: str = "cpu") -> None:
    print("=" * 60)
    print("  SafeFall Interactive Demo")
    print(f"  device: {device}")
    print("=" * 60)
    print()

    print("[1/4] Loading velocity policy ...")
    vel_fn = load_velocity_policy(device)

    print("[2/4] Loading SafeFall policy ...")
    safefall_fn = load_safefall_policy(device)

    print("[3/4] Loading HoST policy ...")
    host_fn = load_host_policy(device)

    print("[4/4] Loading fall predictor ...")
    predictor = load_predictor_wrapper(device)
    print()

    print("Ready — press A to push, B to stand up, Esc to quit.")
    print()

    run_interactive(
        device=device,
        vel_fn=vel_fn,
        safefall_fn=safefall_fn,
        host_fn=host_fn,
        predictor=predictor,
        env_cfg_vel=lambda play: load_env_cfg(_VELOCITY_TASK, play=play),
        env_cfg_safefall=lambda play: load_env_cfg(_SAFEFALL_TASK, play=play),
        env_cfg_host=lambda play: load_env_cfg(_HOST_TASK, play=play),
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SafeFall demo")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    main(device=args.device)
