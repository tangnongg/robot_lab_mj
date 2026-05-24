"""Unitree B2 base roll-pitch-yaw and height tracking task."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from robot_lab.asset_zoo.robots import get_unitree_b2_robot_cfg

from . import mdp
from .mdp import UniformRpyBaseHeightCommandCfg


def unitree_b2_flat_rpy_bh_track_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a flat-ground task for tracking base orientation and height."""

  joint_asset_cfg = SceneEntityCfg("robot", joint_names=(".*_joint",))
  foot_geom_names = (
    "FL_foot_collision",
    "FR_foot_collision",
    "RL_foot_collision",
    "RR_foot_collision",
  )

  feet_ground_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  nonfeet_ground_contact = ContactSensorCfg(
    name="nonfeet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision\d*$",
      entity="robot",
      exclude=foot_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )

  actor_terms = {
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "base_pose"},
    ),
    "pose_error": ObservationTermCfg(
      func=mdp.pose_command_error,
      params={"command_name": "base_pose"},
    ),
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": joint_asset_cfg},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": joint_asset_cfg},
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic_terms = {
    **actor_terms,
    "base_rpy": ObservationTermCfg(func=mdp.base_rpy),
    "base_height": ObservationTermCfg(func=mdp.base_height),
  }

  commands: dict[str, CommandTermCfg] = {
    "base_pose": UniformRpyBaseHeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 6.0),
      debug_vis=False,
      ranges=UniformRpyBaseHeightCommandCfg.Ranges(
        roll=(-0.30, 0.30),
        pitch=(-0.30, 0.30),
        yaw=(-0.60, 0.60),
        base_height=(0.52, 0.64),
      ),
    )
  }

  rewards = {
    "track_base_orientation_exp": RewardTermCfg(
      func=mdp.track_base_orientation_exp,
      weight=5.0,
      params={"command_name": "base_pose", "std": 0.35},
    ),
    "track_base_height_exp": RewardTermCfg(
      func=mdp.track_base_height_exp,
      weight=3.0,
      params={"command_name": "base_pose", "std": 0.05},
    ),
    "feet_contact_count_exp": RewardTermCfg(
      func=mdp.feet_contact_count_exp,
      weight=0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "expected_contacts": 4,
        "contact_threshold": 1.0,
        "std": 1.0,
      },
    ),
    "base_xy_position_l2": RewardTermCfg(
      func=mdp.base_xy_position_l2,
      weight=-0.5,
    ),
    "base_lin_vel_xy_l2": RewardTermCfg(
      func=mdp.base_lin_vel_xy_l2,
      weight=-1.0,
    ),
    "base_lin_vel_z_l2": RewardTermCfg(
      func=mdp.base_lin_vel_z_l2,
      weight=-2.0,
    ),
    "base_ang_vel_l2": RewardTermCfg(
      func=mdp.base_ang_vel_l2,
      weight=-0.05,
    ),
    "joint_torques_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-2.5e-5,
      params={"asset_cfg": joint_asset_cfg},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": joint_asset_cfg},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": joint_asset_cfg},
    ),
    "action_rate_l2": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.01,
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-1.0,
      params={"sensor_name": "nonfeet_ground_contact", "threshold": 1.0},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    sim=SimulationCfg(
      nconmax=128,
      njmax=1024,
      contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
        impratio=10,
        cone="elliptic",
      ),
    ),
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_unitree_b2_robot_cfg()},
      sensors=(feet_ground_contact, nonfeet_ground_contact),
      num_envs=1,
      extent=2.0,
    ),
    episode_length_s=20.0,
    is_finite_horizon=False,
    scale_rewards_by_dt=True,
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=False,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
      ),
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale={
          r".*_hip_joint": 0.15,
          r"^(?!.*_hip_joint).*": 0.25,
        },
        use_default_offset=True,
        clip={r".*": (-100.0, 100.0)},
      )
    },
    commands=commands,
    events={
      "randomize_reset_joints": EventTermCfg(
        func=envs_mdp.reset_joints_by_offset,
        mode="reset",
        params={
          "position_range": (-0.10, 0.10),
          "velocity_range": (-0.20, 0.20),
          "asset_cfg": joint_asset_cfg,
        },
      ),
      "randomize_reset_base": EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "pose_range": {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.10, 0.10),
          },
          "velocity_range": {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (-0.10, 0.10),
            "roll": (-0.10, 0.10),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.10, 0.10),
          },
        },
      ),
      "push_robot": EventTermCfg(
        func=envs_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "velocity_range": {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
            "yaw": (-0.25, 0.25),
          },
        },
      ),
    },
    rewards=rewards,
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "bad_orientation": TerminationTermCfg(
        func=envs_mdp.bad_orientation,
        params={"limit_angle": math.radians(75.0)},
      ),
      "base_height_too_low": TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum,
        params={"minimum_height": 0.35},
      ),
      "illegal_contact": TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "nonfeet_ground_contact", "force_threshold": 1.0},
      ),
      "nan_term": TerminationTermCfg(
        func=envs_mdp.nan_detection,
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    seed=None,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.events.pop("push_robot", None)

  return cfg
