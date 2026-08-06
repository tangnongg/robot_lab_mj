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
    parser.add_argument("--success-hold-time", type=float, default=5.0)
    parser.add_argument("--max-angular-speed", type=float, default=0.5)
    parser.add_argument("--max-linear-speed", type=float, default=0.5)
    parser.add_argument("--max-feet-speed", type=float, default=0.2)
    parser.add_argument("--max-joint-speed", type=float, default=0.2)
    parser.add_argument("--max-vertical-speed", type=float, default=0.1)
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
    foot_ids = [
        body_names.index("left_ankle_roll_link"),
        body_names.index("right_ankle_roll_link"),
    ]
    joint_ids = base_env.action_manager.get_term("joint_pos").target_ids
    site_names = list(asset.site_names)
    head_site_id = site_names.index("head_link")
    left_ankle_site_ids = [
        site_names.index(f"left_ankle_keypoint_{name}")
        for name in ("center", "front", "back", "left", "right")
    ]
    right_ankle_site_ids = [
        site_names.index(f"right_ankle_keypoint_{name}")
        for name in ("center", "front", "back", "left", "right")
    ]

    obs = env.get_observations()
    max_base_height = torch.zeros(args.num_envs, device=base_env.device)
    max_head_height = torch.zeros_like(max_base_height)
    upright_steps = torch.zeros(args.num_envs, dtype=torch.long, device=base_env.device)
    best_upright_steps = torch.zeros_like(upright_steps)
    stable_steps = torch.zeros_like(upright_steps)
    best_stable_steps = torch.zeros_like(upright_steps)
    ankle_upright_steps = torch.zeros_like(upright_steps)
    ankle_parallel_steps = torch.zeros_like(upright_steps)
    ankle_var_sum = torch.zeros_like(max_base_height)
    ankle_var_max = torch.zeros_like(max_base_height)
    ang_speed_sum = torch.zeros_like(max_base_height)
    yaw_speed_sum = torch.zeros_like(max_base_height)
    base_xy_speed_sum = torch.zeros_like(max_base_height)
    feet_xy_speed_sum = torch.zeros_like(max_base_height)
    joint_speed_sum = torch.zeros_like(max_base_height)
    vertical_speed_sum = torch.zeros_like(max_base_height)
    completed: list[tuple[float, ...]] = []
    trace_steps = {0, 1, 30, 31, 50, 100, 200, 300, 400, 499}
    step = 0

    while len(completed) < args.episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        base_height = asset.data.root_link_pos_w[:, 2]
        head_height = asset.data.site_pos_w[:, head_site_id, 2]
        gravity_z = asset.data.projected_gravity_b[:, 2]

        max_base_height.copy_(torch.maximum(max_base_height, base_height))
        max_head_height.copy_(torch.maximum(max_head_height, head_height))
        upright = (base_height > 0.65) & (gravity_z < -0.9)
        left_z = asset.data.site_pos_w[:, left_ankle_site_ids, 2] * 10.0
        right_z = asset.data.site_pos_w[:, right_ankle_site_ids, 2] * 10.0
        ankle_var = (left_z.var(dim=1) + right_z.var(dim=1)) / 2.0
        ankle_upright_steps += upright.long()
        ankle_parallel_steps += (upright & (ankle_var < 0.05)).long()
        ankle_var_sum += torch.where(upright, ankle_var, 0.0)
        ankle_var_max.copy_(
            torch.where(
                upright, torch.maximum(ankle_var_max, ankle_var), ankle_var_max
            )
        )
        root_ang_vel = asset.data.root_link_ang_vel_w
        root_lin_vel = asset.data.root_link_lin_vel_w
        feet_lin_vel = asset.data.body_link_lin_vel_w[:, foot_ids]
        ang_speed = torch.linalg.vector_norm(root_ang_vel, dim=1)
        yaw_speed = torch.abs(root_ang_vel[:, 2])
        base_xy_speed = torch.linalg.vector_norm(root_lin_vel[:, :2], dim=1)
        vertical_speed = torch.abs(root_lin_vel[:, 2])
        joint_speed = torch.amax(
            torch.abs(asset.data.joint_vel[:, joint_ids]), dim=1
        )
        feet_xy_speed = torch.linalg.vector_norm(
            feet_lin_vel[..., :2], dim=2
        ).mean(dim=1)
        upright_steps = torch.where(upright, upright_steps + 1, 0)
        best_upright_steps.copy_(torch.maximum(best_upright_steps, upright_steps))
        stable = (
            upright
            & (ang_speed < args.max_angular_speed)
            & (base_xy_speed < args.max_linear_speed)
            & (feet_xy_speed < args.max_feet_speed)
            & (joint_speed < args.max_joint_speed)
            & (vertical_speed < args.max_vertical_speed)
        )
        stable_steps = torch.where(stable, stable_steps + 1, 0)
        best_stable_steps.copy_(torch.maximum(best_stable_steps, stable_steps))
        ang_speed_sum += torch.where(upright, ang_speed, 0.0)
        yaw_speed_sum += torch.where(upright, yaw_speed, 0.0)
        base_xy_speed_sum += torch.where(upright, base_xy_speed, 0.0)
        feet_xy_speed_sum += torch.where(upright, feet_xy_speed, 0.0)
        joint_speed_sum += torch.where(upright, joint_speed, 0.0)
        vertical_speed_sum += torch.where(upright, vertical_speed, 0.0)

        if args.trace and step in trace_steps:
            beta = getattr(base_env, "host_action_rescale")[0].item()
            force = getattr(base_env, "host_traction_force")[0].item()
            print(
                f"trace step={step:04d} base_z={base_height[0].item():.3f} "
                f"head_z={head_height[0].item():.3f} "
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
            stable_time = best_stable_steps[env_id].item() * base_env.step_dt
            upright_count = max(ankle_upright_steps[env_id].item(), 1)
            completed.append(
                (
                    hold_time >= args.success_hold_time,
                    stable_time >= args.success_hold_time,
                    max_base_height[env_id].item(),
                    max_head_height[env_id].item(),
                    hold_time,
                    stable_time,
                    ankle_parallel_steps[env_id].item() / upright_count,
                    ankle_var_sum[env_id].item() / upright_count,
                    ankle_var_max[env_id].item(),
                    ang_speed_sum[env_id].item() / upright_count,
                    yaw_speed_sum[env_id].item() / upright_count,
                    base_xy_speed_sum[env_id].item() / upright_count,
                    feet_xy_speed_sum[env_id].item() / upright_count,
                    joint_speed_sum[env_id].item() / upright_count,
                    vertical_speed_sum[env_id].item() / upright_count,
                )
            )
            if len(completed) >= args.episodes:
                break
        if len(done_ids) > 0:
            max_base_height[done_ids] = 0.0
            max_head_height[done_ids] = 0.0
            upright_steps[done_ids] = 0
            best_upright_steps[done_ids] = 0
            stable_steps[done_ids] = 0
            best_stable_steps[done_ids] = 0
            ankle_upright_steps[done_ids] = 0
            ankle_parallel_steps[done_ids] = 0
            ankle_var_sum[done_ids] = 0.0
            ankle_var_max[done_ids] = 0.0
            ang_speed_sum[done_ids] = 0.0
            yaw_speed_sum[done_ids] = 0.0
            base_xy_speed_sum[done_ids] = 0.0
            feet_xy_speed_sum[done_ids] = 0.0
            joint_speed_sum[done_ids] = 0.0
            vertical_speed_sum[done_ids] = 0.0
        step += 1

    result = torch.tensor(completed[: args.episodes])
    print(f"episodes={args.episodes}")
    print(f"success_rate={result[:, 0].mean().item():.4f}")
    print(f"stable_success_rate={result[:, 1].mean().item():.4f}")
    print(f"mean_max_base_height={result[:, 2].mean().item():.4f}")
    print(f"mean_max_head_height={result[:, 3].mean().item():.4f}")
    print(f"mean_best_upright_time={result[:, 4].mean().item():.4f}")
    print(f"mean_best_stable_time={result[:, 5].mean().item():.4f}")
    print(f"upright_ankle_parallel_rate={result[:, 6].mean().item():.4f}")
    print(f"upright_mean_ankle_variance={result[:, 7].mean().item():.6f}")
    print(f"upright_mean_max_ankle_variance={result[:, 8].mean().item():.6f}")
    print(f"upright_mean_angular_speed={result[:, 9].mean().item():.6f}")
    print(f"upright_mean_yaw_speed={result[:, 10].mean().item():.6f}")
    print(f"upright_mean_base_xy_speed={result[:, 11].mean().item():.6f}")
    print(f"upright_mean_feet_xy_speed={result[:, 12].mean().item():.6f}")
    print(f"upright_mean_joint_speed={result[:, 13].mean().item():.6f}")
    print(f"upright_mean_vertical_speed={result[:, 14].mean().item():.6f}")
    env.close()


if __name__ == "__main__":
    main()
