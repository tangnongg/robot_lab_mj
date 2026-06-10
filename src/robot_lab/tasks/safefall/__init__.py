"""SafeFall task for Unitree G1 humanoid (migrated from IsaacLab).

Register the SafeFall-G1 environment with the mjlab task registry.
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_safefall_env_cfg
from .rl_cfg import unitree_g1_safefall_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-SafeFall-G1",
    env_cfg=unitree_g1_safefall_env_cfg(),
    play_env_cfg=unitree_g1_safefall_env_cfg(play=True),
    rl_cfg=unitree_g1_safefall_ppo_runner_cfg(),
)
