"""Headless evaluation for the HoST supine stand-up task."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401 - populate the task registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


TASK_ID = "Mjlab-HoST-Ground-Unitree-G1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-step", type=int, default=50)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--force", type=float, default=0.0)
    parser.add_argument("--action-rescale", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 10.0
    env_cfg.seed = args.seed

    base_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=args.device,
        render_mode="rgb_array" if args.snapshot is not None else None,
    )
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    base_env.host_traction_force.fill_(args.force)
    base_env.host_action_rescale.fill_(args.action_rescale)

    if args.checkpoint is None:
        policy = lambda obs: torch.zeros(  # noqa: E731
            base_env.action_space.shape, device=base_env.device
        )
    else:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(
            str(args.checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)

    asset = base_env.scene["robot"]
    body_names = list(asset.body_names)
    torso_idx = body_names.index("torso_link")
    foot_ids = [
        body_names.index("left_ankle_roll_link"),
        body_names.index("right_ankle_roll_link"),
    ]

    obs = env.get_observations()
    max_base_height = torch.zeros(args.num_envs, device=base_env.device)
    max_head_height = torch.zeros_like(max_base_height)
    upright_steps = torch.zeros(args.num_envs, dtype=torch.long, device=base_env.device)
    best_upright_steps = torch.zeros_like(upright_steps)
    completed: list[tuple[bool, float, float, float]] = []
    trace_steps = {0, 1, 30, 31, 50, 100, 200, 300, 400, 499}
    step = 0

    while len(completed) < args.episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        base_height = asset.data.root_link_pos_w[:, 2]
        head_height = asset.data.body_link_pos_w[:, torso_idx, 2]
        feet_height = asset.data.body_link_pos_w[:, foot_ids, 2].mean(dim=-1)
        relative_head_height = head_height - feet_height
        gravity_z = asset.data.projected_gravity_b[:, 2]

        max_base_height.copy_(torch.maximum(max_base_height, base_height))
        max_head_height.copy_(torch.maximum(max_head_height, relative_head_height))
        upright = (base_height > 0.65) & (gravity_z < -0.9)
        upright_steps = torch.where(upright, upright_steps + 1, 0)
        best_upright_steps.copy_(torch.maximum(best_upright_steps, upright_steps))

        if args.trace and step in trace_steps:
            beta = getattr(base_env, "host_action_rescale")[0].item()
            force = getattr(base_env, "host_traction_force")[0].item()
            print(
                f"trace step={step:04d} base_z={base_height[0].item():.3f} "
                f"head_rel={relative_head_height[0].item():.3f} "
                f"gravity_z={gravity_z[0].item():.3f} force={force:.1f} beta={beta:.2f}",
                flush=True,
            )

        if args.snapshot is not None and step == args.snapshot_step:
            from PIL import Image

            frame = base_env.render()
            assert frame is not None
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(args.snapshot)
            if args.snapshot_only:
                print(f"snapshot={args.snapshot}")
                env.close()
                return

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        for env_id in done_ids.tolist():
            hold_time = best_upright_steps[env_id].item() * base_env.step_dt
            completed.append(
                (
                    hold_time >= 0.5,
                    max_base_height[env_id].item(),
                    max_head_height[env_id].item(),
                    hold_time,
                )
            )
            if len(completed) >= args.episodes:
                break
        if len(done_ids) > 0:
            max_base_height[done_ids] = 0.0
            max_head_height[done_ids] = 0.0
            upright_steps[done_ids] = 0
            best_upright_steps[done_ids] = 0
        step += 1

    result = torch.tensor(completed[: args.episodes])
    print(f"episodes={args.episodes}")
    print(f"success_rate={result[:, 0].mean().item():.4f}")
    print(f"mean_max_base_height={result[:, 1].mean().item():.4f}")
    print(f"mean_max_head_height={result[:, 2].mean().item():.4f}")
    print(f"mean_best_upright_time={result[:, 3].mean().item():.4f}")
    env.close()


if __name__ == "__main__":
    main()
