"""Record a headless HoST rollout to an MP4 file."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401 - populate the task registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder


TASK_ID = "Mjlab-HoST-Ground-Unitree-G1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", type=float, default=0.0)
    parser.add_argument("--action-rescale", type=float, default=0.25)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the policy mean instead of the stochastic PPO action sampler.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s, args.frames * env_cfg.decimation * env_cfg.sim.mujoco.timestep
    )
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
    video_dir = args.output.parent if args.output is not None else args.checkpoint.parent / "videos" / "play"
    video_name = args.output.name if args.output is not None else "host-rollout.mp4"
    video_env = VideoRecorder(
        base_env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.frames,
        name_prefix=Path(video_name).stem,
        disable_logger=False,
    )

    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
    # Curriculum and action buffers are created during wrapper initialization.
    base_env.host_traction_force.fill_(args.force)
    base_env.host_action_rescale.fill_(args.action_rescale)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    try:
        obs = env.get_observations()
        for _ in range(args.frames):
            with torch.inference_mode():
                # Keep recorded rollouts consistent with PPO's training-time
                # action sampling unless deterministic mode is requested.
                obs, *_ = env.step(
                    policy(obs, stochastic_output=not args.deterministic)
                )
    finally:
        env.close()

    output_path = video_dir / f"{Path(video_name).stem}-step-0.mp4"
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Video was not written: {output_path}")
    print(f"video={output_path}")


if __name__ == "__main__":
    main()
