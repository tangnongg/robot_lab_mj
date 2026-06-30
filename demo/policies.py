"""Policy loading for the SafeFall demo."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from robot_lab.tasks.safefall.fall_predictor import load_predictor

# ── Checkpoint paths ──────────────────────────────────────────

VELOCITY_CKPT = "logs/rsl_rl/g1_velocity/2026-06-11_01-03-02/model_1999.pt"
SAFEFALL_CKPT = "logs/rsl_rl/safefall_g1/2026-06-30_07-29-09/model_4998.pt"
HOST_CKPT = "logs/rsl_rl/g1_host_ground/2026-06-09_05-00-56/model_900.pt"
PREDICTOR_CKPT = "models/fall_predictor.pt"

VELOCITY_TASK = "Mjlab-Velocity-Flat-Unitree-G1"
SAFEFALL_TASK = "Mjlab-SafeFall-G1"
HOST_TASK = "Mjlab-HoST-Ground-Unitree-G1"


def load_velocity_policy(device: str) -> Callable:
    """Load velocity-tracking policy. Returns ``fn(obs_dict) → action``."""
    return _load(VELOCITY_TASK, VELOCITY_CKPT, device)


def load_safefall_policy(device: str) -> Callable:
    """Load SafeFall mitigation policy."""
    return _load(SAFEFALL_TASK, SAFEFALL_CKPT, device)


def load_host_policy(device: str) -> Callable:
    """Load HoST stand-up policy."""
    return _load(HOST_TASK, HOST_CKPT, device)


def load_predictor_wrapper(device: str, env=None):
    """Load fall predictor with deployment wrapper."""
    return load_predictor(PREDICTOR_CKPT, threshold=0.5, device=device,
                          warmup_steps=10, confirmation_steps=3, env=env)


# ── internals ──────────────────────────────────────────────────

def _load(task_id: str, ckpt: str, device: str) -> Callable:
    rl_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)
    runner = runner_cls(wrapped, asdict(rl_cfg), log_dir="/tmp", device=device)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    env.close()

    def _fn(obs_dict):
        return policy(obs_dict)

    return _fn
