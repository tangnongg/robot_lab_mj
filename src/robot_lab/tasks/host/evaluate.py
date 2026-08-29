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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the policy mean instead of the stochastic PPO action sampler.",
    )
    parser.add_argument("--diagnostics", action="store_true")
    return parser.parse_args()


def _foot_slip_cost(base_env: ManagerBasedRlEnv, asset) -> torch.Tensor:
    """Horizontal velocity squared for each currently contacting ankle."""
    sensor = base_env.scene["feet_ground_contact"]
    found = sensor.data.found
    if found is None:
        return torch.zeros(base_env.num_envs, device=base_env.device)
    if found.ndim == 3:
        found = found.squeeze(-1)
    names = [str(name).lower() for name in sensor.primary_names]
    left_ids = [i for i, name in enumerate(names) if "left_foot" in name]
    right_ids = [i for i, name in enumerate(names) if "right_foot" in name]
    if left_ids and right_ids:
        contact = torch.stack(
            ((found[:, left_ids] > 0).any(dim=1), (found[:, right_ids] > 0).any(dim=1)),
            dim=1,
        )
    else:
        contact = (found > 0).any(dim=1, keepdim=True).expand(-1, 2)
    body_names = list(asset.body_names)
    foot_ids = [
        body_names.index("left_ankle_roll_link"),
        body_names.index("right_ankle_roll_link"),
    ]
    foot_vel = asset.data.body_link_lin_vel_w[:, foot_ids, :2]
    return (torch.square(foot_vel).sum(dim=-1) * contact.float()).sum(dim=1)


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
    obs = env.get_observations()
    max_base_height = torch.zeros(args.num_envs, device=base_env.device)
    max_head_height = torch.zeros_like(max_base_height)
    upright_steps = torch.zeros(args.num_envs, dtype=torch.long, device=base_env.device)
    best_upright_steps = torch.zeros_like(upright_steps)
    tail_steps = torch.zeros(args.num_envs, device=base_env.device)
    tail_ang_sq = torch.zeros_like(tail_steps)
    tail_lin_sq = torch.zeros_like(tail_steps)
    tail_joint_sq = torch.zeros_like(tail_steps)
    tail_slip_sq = torch.zeros_like(tail_steps)
    phase_reached = torch.zeros(args.num_envs, dtype=torch.bool, device=base_env.device)
    post_task_entered = torch.zeros_like(phase_reached)
    completed: list[tuple[bool, bool, bool, float, float, float]] = []
    diagnostics: list[tuple[float, float, float, float]] = []
    trace_steps = {0, 1, 30, 31, 50, 100, 200, 300, 400, 499}
    step = 0

    while len(completed) < args.episodes:
        with torch.inference_mode():
            # Match PPO training: actions are sampled from the policy unless
            # deterministic evaluation is explicitly requested.
            actions = policy(obs, stochastic_output=not args.deterministic)
            obs, _, dones, extras = env.step(actions)

        base_height = asset.data.root_link_pos_w[:, 2]
        head_idx = list(asset.site_names).index("head_link")
        # Match the training reward/event definition exactly: world-frame
        # height of the head_link site, not height relative to the feet.
        head_height = asset.data.site_pos_w[:, head_idx, 2]
        gravity_z = asset.data.projected_gravity_b[:, 2]

        max_base_height.copy_(torch.maximum(max_base_height, base_height))
        max_head_height.copy_(torch.maximum(max_head_height, head_height))
        # Match the training state machine: three consecutive head-height /
        # orientation frames mark stand-up reached, followed by a ten-frame
        # transition before POST_TASK becomes active.
        current_phase = getattr(base_env, "host_standup_phase", None)
        current_reached = getattr(base_env, "host_standup_reached", None)
        if current_phase is not None:
            phase_reached |= current_phase >= 1
            post_task_entered |= current_phase == 2
        if current_reached is not None:
            phase_reached |= current_reached
        upright = (head_height > 1.27) & (gravity_z < -0.55)
        upright_steps = torch.where(upright, upright_steps + 1, 0)
        best_upright_steps.copy_(torch.maximum(best_upright_steps, upright_steps))
        if args.diagnostics:
            # Match the training gate: exclude the first 0.5 s after the
            # upright condition, when the robot is still settling into contact.
            mask = (upright & (upright_steps >= 25)).float()
            tail_steps += mask
            tail_ang_sq += torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1) * mask
            tail_lin_sq += torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1) * mask
            tail_joint_sq += torch.mean(torch.square(asset.data.joint_vel), dim=1) * mask
            tail_slip_sq += _foot_slip_cost(base_env, asset) * mask

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
            completed.append((
                bool(phase_reached[env_id].item()),
                bool(post_task_entered[env_id].item()),
                bool(hold_time >= 0.5),
                max_base_height[env_id].item(),
                max_head_height[env_id].item(),
                hold_time,
            ))
            if args.diagnostics:
                denom = tail_steps[env_id].clamp(min=1.0)
                diagnostics.append(
                    (
                        torch.sqrt(tail_ang_sq[env_id] / denom).item(),
                        torch.sqrt(tail_lin_sq[env_id] / denom).item(),
                        torch.sqrt(tail_joint_sq[env_id] / denom).item(),
                        torch.sqrt(tail_slip_sq[env_id] / denom).item(),
                    )
                )
            if len(completed) >= args.episodes:
                break
        if len(done_ids) > 0:
            max_base_height[done_ids] = 0.0
            max_head_height[done_ids] = 0.0
            upright_steps[done_ids] = 0
            best_upright_steps[done_ids] = 0
            phase_reached[done_ids] = False
            post_task_entered[done_ids] = False
            tail_steps[done_ids] = 0.0
            tail_ang_sq[done_ids] = 0.0
            tail_lin_sq[done_ids] = 0.0
            tail_joint_sq[done_ids] = 0.0
            tail_slip_sq[done_ids] = 0.0
        step += 1

    result = torch.tensor(completed[: args.episodes])
    print(f"episodes={args.episodes}")
    print(f"standup_reached_rate={result[:, 0].mean().item():.4f}")
    print(f"post_task_entry_rate={result[:, 1].mean().item():.4f}")
    print(f"quiet_hold_success_rate={result[:, 2].mean().item():.4f}")
    print(f"mean_max_base_height={result[:, 3].mean().item():.4f}")
    print(f"mean_max_head_height={result[:, 4].mean().item():.4f}")
    print(f"mean_best_upright_time={result[:, 5].mean().item():.4f}")
    if args.diagnostics and diagnostics:
        diag = torch.tensor(diagnostics[: args.episodes])
        print(f"mean_post_root_ang_vel_rms={diag[:, 0].mean().item():.6f}")
        print(f"mean_post_root_lin_vel_rms={diag[:, 1].mean().item():.6f}")
        print(f"mean_post_joint_vel_rms={diag[:, 2].mean().item():.6f}")
        print(f"mean_post_foot_slip_vel_rms={diag[:, 3].mean().item():.6f}")
    env.close()


if __name__ == "__main__":
    main()
