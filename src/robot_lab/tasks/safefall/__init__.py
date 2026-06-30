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

# Stage II curriculum: sample initial states from the predictor-flagged
# bank (requires running prepare_stage2_states.py first).
register_mjlab_task(
    task_id="Mjlab-SafeFall-StageII-G1",
    env_cfg=unitree_g1_safefall_env_cfg(stage2=True),
    play_env_cfg=unitree_g1_safefall_env_cfg(play=True, stage2=True),
    rl_cfg=unitree_g1_safefall_ppo_runner_cfg(),
)
