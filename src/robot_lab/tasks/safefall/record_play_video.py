#!/usr/bin/env python3
"""Record a deterministic SafeFall policy rollout without an interactive viewer."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# CI and headless training hosts generally do not provide X11.  Set this
# before importing MuJoCo through mjlab so recording always uses EGL offscreen
# rendering unless the caller explicitly selected another backend.
os.environ.setdefault("MUJOCO_GL", "egl")

_src = Path(__file__).resolve().parents[4]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import robot_lab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder


def record(
    checkpoint: str | Path,
    *,
    frames: int = 200,
    width: int = 1280,
    height: int = 720,
    device: str = "cuda:0",
) -> Path:
    """Record ``frames`` deterministic policy steps and return the MP4 path."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    task_id = "Mjlab-SafeFall-G1"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    # Keep one fall in view for the entire recording so post-impact settling is
    # observable instead of being hidden by the training horizon reset.
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        frames * env_cfg.decimation * env_cfg.sim.mujoco.timestep,
    )
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height

    video_dir = checkpoint.parent / "videos" / "play"
    base_env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
    video_env = VideoRecorder(
        base_env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=frames,
        disable_logger=False,
    )
    agent_cfg = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    try:
        for _ in range(frames):
            with torch.inference_mode():
                env.step(policy(env.get_observations()))
    finally:
        env.close()

    video_path = video_dir / "rl-video-step-0.mp4"
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Video was not written: {video_path}")
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    path = record(
        args.checkpoint,
        frames=args.frames,
        width=args.width,
        height=args.height,
        device=args.device,
    )
    print(path)


if __name__ == "__main__":
    main()
