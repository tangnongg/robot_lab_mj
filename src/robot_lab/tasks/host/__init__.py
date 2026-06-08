"""HoST stand-up task registration for the Unitree G1 humanoid.

Five environment variants (ground, platform, wall, slope, prone),
each with train and play configurations.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    unitree_g1_ground_env_cfg,
    unitree_g1_platform_env_cfg,
    unitree_g1_prone_env_cfg,
    unitree_g1_slope_env_cfg,
    unitree_g1_wall_env_cfg,
)
from .rl_cfg import (
    unitree_g1_host_ground_ppo_runner_cfg,
    unitree_g1_host_platform_ppo_runner_cfg,
    unitree_g1_host_prone_ppo_runner_cfg,
    unitree_g1_host_slope_ppo_runner_cfg,
    unitree_g1_host_wall_ppo_runner_cfg,
)

# ---- Ground ----
register_mjlab_task(
    task_id="Mjlab-HoST-Ground-Unitree-G1",
    env_cfg=unitree_g1_ground_env_cfg(),
    play_env_cfg=unitree_g1_ground_env_cfg(play=True),
    rl_cfg=unitree_g1_host_ground_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# ---- Platform ----
register_mjlab_task(
    task_id="Mjlab-HoST-Platform-Unitree-G1",
    env_cfg=unitree_g1_platform_env_cfg(),
    play_env_cfg=unitree_g1_platform_env_cfg(play=True),
    rl_cfg=unitree_g1_host_platform_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# ---- Wall ----
register_mjlab_task(
    task_id="Mjlab-HoST-Wall-Unitree-G1",
    env_cfg=unitree_g1_wall_env_cfg(),
    play_env_cfg=unitree_g1_wall_env_cfg(play=True),
    rl_cfg=unitree_g1_host_wall_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# ---- Slope ----
register_mjlab_task(
    task_id="Mjlab-HoST-Slope-Unitree-G1",
    env_cfg=unitree_g1_slope_env_cfg(),
    play_env_cfg=unitree_g1_slope_env_cfg(play=True),
    rl_cfg=unitree_g1_host_slope_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# ---- Prone ----
register_mjlab_task(
    task_id="Mjlab-HoST-Prone-Unitree-G1",
    env_cfg=unitree_g1_prone_env_cfg(),
    play_env_cfg=unitree_g1_prone_env_cfg(play=True),
    rl_cfg=unitree_g1_host_prone_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
