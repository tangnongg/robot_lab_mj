from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_b2_flat_rpy_bh_track_env_cfg
from .rl_cfg import unitree_b2_rpy_bh_track_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-RpyBhTrack-Flat-Unitree-B2",
  env_cfg=unitree_b2_flat_rpy_bh_track_env_cfg(),
  play_env_cfg=unitree_b2_flat_rpy_bh_track_env_cfg(play=True),
  rl_cfg=unitree_b2_rpy_bh_track_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
