"""HoST stand-up task registration for the Unitree G1 humanoid.

Five environment variants (ground, platform, wall, slope, prone),
each with train and play configurations.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner

from .env_cfgs import (
    unitree_g1_ground_env_cfg,
    unitree_g1_ground_standup_env_cfg,
    unitree_g1_ground_hold_env_cfg,
    unitree_g1_platform_env_cfg,
    unitree_g1_prone_env_cfg,
    unitree_g1_slope_env_cfg,
    unitree_g1_wall_env_cfg,
)
from .rl_cfg import (
    unitree_g1_host_ground_ppo_runner_cfg,
    unitree_g1_host_ground_standup_only_ppo_runner_cfg,
    unitree_g1_host_platform_ppo_runner_cfg,
    unitree_g1_host_prone_ppo_runner_cfg,
    unitree_g1_host_slope_ppo_runner_cfg,
    unitree_g1_host_wall_ppo_runner_cfg,
)

# ---- Canonical end-to-end ground task ----
register_mjlab_task(
    task_id="Mjlab-HoST-Ground-Unitree-G1",
    env_cfg=unitree_g1_ground_env_cfg(),
    play_env_cfg=unitree_g1_ground_env_cfg(play=True),
    rl_cfg=unitree_g1_host_ground_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

# Historical task IDs below are aliases to the same complete task.  They are
# kept only so old scripts do not silently select a different objective.
register_mjlab_task(
    task_id="Mjlab-HoST-Ground-StandupOnly-Unitree-G1",
    env_cfg=unitree_g1_ground_standup_env_cfg(),
    play_env_cfg=unitree_g1_ground_standup_env_cfg(play=True),
    rl_cfg=unitree_g1_host_ground_standup_only_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-HoST-Ground-StandupHold-Unitree-G1",
    env_cfg=unitree_g1_ground_hold_env_cfg(),
    play_env_cfg=unitree_g1_ground_hold_env_cfg(play=True),
    rl_cfg=unitree_g1_host_ground_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

# # ---- Platform ----
# register_mjlab_task(
#     task_id="Mjlab-HoST-Platform-Unitree-G1",
#     env_cfg=unitree_g1_platform_env_cfg(),
#     play_env_cfg=unitree_g1_platform_env_cfg(play=True),
#     rl_cfg=unitree_g1_host_platform_ppo_runner_cfg(),
#     runner_cls=VelocityOnPolicyRunner,
# )

# # ---- Wall ----
# register_mjlab_task(
#     task_id="Mjlab-HoST-Wall-Unitree-G1",
#     env_cfg=unitree_g1_wall_env_cfg(),
#     play_env_cfg=unitree_g1_wall_env_cfg(play=True),
#     rl_cfg=unitree_g1_host_wall_ppo_runner_cfg(),
#     runner_cls=VelocityOnPolicyRunner,
# )

# # ---- Slope ----
# register_mjlab_task(
#     task_id="Mjlab-HoST-Slope-Unitree-G1",
#     env_cfg=unitree_g1_slope_env_cfg(),
#     play_env_cfg=unitree_g1_slope_env_cfg(play=True),
#     rl_cfg=unitree_g1_host_slope_ppo_runner_cfg(),
#     runner_cls=VelocityOnPolicyRunner,
# )

# # ---- Prone ----
# register_mjlab_task(
#     task_id="Mjlab-HoST-Prone-Unitree-G1",
#     env_cfg=unitree_g1_prone_env_cfg(),
#     play_env_cfg=unitree_g1_prone_env_cfg(play=True),
#     rl_cfg=unitree_g1_host_prone_ppo_runner_cfg(),
#     runner_cls=VelocityOnPolicyRunner,
# )
