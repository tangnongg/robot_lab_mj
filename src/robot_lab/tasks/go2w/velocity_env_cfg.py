# "Velocity task configuration.

# This module provides a factory function to create a base velocity task config.
# Robot-specific configurations call the factory and customize as needed.
# "

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  ContactMatch,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
  ContactSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from robot_lab.tasks.go2w import mdp
from robot_lab.tasks.go2w.mdp import UniformThresholdVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from robot_lab.terrains.rough import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  "Create base velocity tracking task configuration."

  ##
  # Sensors
  ##

  terrain_scanner = RayCastSensorCfg(
    name="terrain_scanner",
    frame=ObjRef(type="site", name="terrain_scan_site", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    # viz default: 
        # hit: green
        # miss: red
        # hit sphere: cyan
  )

  height_scanner_base = RayCastSensorCfg(
    name="height_scanner_base",
    frame=ObjRef(type="site", name="height_scanner_base_site", entity="robot"),  # Set per-robot: frame and pattern.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.1, 0.1), resolution=0.05),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ## contact_forces = ContactSensorCfg(

  ##
  # Metrics
  ##

  #TODO: 
  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Commands
  ##

  #TODO: rel_forward_envs needs to be cehcked
  commands: dict[str, CommandTermCfg] = {
    "base_velocity": UniformThresholdVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.02,
      rel_heading_envs=1.0,
      rel_forward_envs=0.0, # in IsaacLab this doesn't exist
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformThresholdVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Observations
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      clip=(-100, 100),
      scale=1.0,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      clip=(-100, 100),
      scale=1.0,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "base_velocity"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scanner"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
  }

  critic_terms = {
    **actor_terms,
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Events
  ##
  events = {
      # ---------- startup ----------
      # TODO: This function creates a set of physics materials with random static friction, 
      # dynamic friction, and restitution values. 
      # "randomize_rigid_body_material": EventTermCfg(
      #     func = mdp.randomize_rigid_body_material,
      #     mode = "startup",
      #     params = {
      #         "asset_cfg": SceneEntityCfg("robot", body_names=((".*")),
      #         "static_friction_range": (0.1, 1.0),
      #         "dynamic_friction_range": (0.1, 0.8),
      #         "restitution_range": (0.0, 0.5),
      #         "num_buckets": 32,
      #     },
      # ),

      # TODO: something different between mjlab and isaaclab, need to check params
      # mass and inertia scale by e^{2\alpha}, there is not add operation
      # body_mass in mjlab has add operation, but it does not recompure inertia
      # Jointly randomize body_mass, body_ipos, body_inertia, and body_iquat
      "randomize_rigid_body_mass_and_inertia": EventTermCfg(
          func = dr.pseudo_inertia,
          mode = "startup",
          params = {
              "asset_cfg": SceneEntityCfg("robot", body_names=()),
              "alpha_range": (-0.1, 0.1), # 0.5
          },
      ),

      ## should included in the above dr.pseudo_inertia
      # "randomize_rigid_body_inertia": EventTermCfg(
      #     func = mdp.randomize_rigid_body_inertia,
      #     mode = "startup",
      #     params = {
      #         "asset_cfg": SceneEntityCfg("robot", body_names=(".*")),
      #         "inertia_distribution_params": (0.5, 1.5),
      #         "operation": "scale",
      #     },
      # ),

      "randomize_com_positions": EventTermCfg(
          func = dr.body_com_offset,
          mode = "startup",
          params = {
              "asset_cfg": SceneEntityCfg("robot", body_names=()),
              "ranges": (-0.1, 0.1),
              "operation": "add",
          },
      ),

      # ---------- reset ----------
      "randomize_apply_external_force_torque": EventTermCfg(
          func = mdp.apply_external_force_torque,
          mode = "reset",
          params = {
              "asset_cfg": SceneEntityCfg("robot", body_names=()),
              "force_range": (-10.0, 10.0),
              "torque_range": (-10.0, 10.0),
          },
      ),

      "randomize_reset_joints": EventTermCfg(
          func = mdp.reset_joints_by_offset,   # 或 reset_joints_by_scale，按需选择
          mode = "reset",
          params = {
              "position_range": (-0.2, 0.2),
              "velocity_range": (-0.5, 0.5),
          },
      ),

      ## "randomize_actuator_gains": EventTermCfg() implemented in dr.joint_stiffness and dr.joint_damping
      "randomize_joint_stiffness": EventTermCfg(
          func = dr.joint_stiffness,
          mode = "reset",
          params = {
              "asset_cfg": SceneEntityCfg("robot", joint_names=(".*")),
              "ranges": (0.5, 2.0),
              "operation": "scale",
              "distribution": "log_uniform",
          },
      ),

      "randomize_joint_damping": EventTermCfg(
          func = dr.joint_damping,
          mode = "reset",
          params = {
              "asset_cfg": SceneEntityCfg("robot", joint_names=(".*")),
              "ranges": (0.5, 2.0),
              "operation": "scale",
              "distribution": "log_uniform",
          },
      ),

      "randomize_reset_base": EventTermCfg(
          func = mdp.reset_root_state_uniform,
          mode = "reset",
          params = {
              "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
              "velocity_range": {
                  "x": (-0.5, 0.5),
                  "y": (-0.5, 0.5),
                  "z": (-0.5, 0.5),
                  "roll": (-0.5, 0.5),
                  "pitch": (-0.5, 0.5),
                  "yaw": (-0.5, 0.5),
              },
          },
      ),

      # ---------- interval ----------
      "randomize_push_robot": EventTermCfg(
          func = mdp.push_by_setting_velocity,
          mode = "interval",
          interval_range_s = (10.0, 15.0),
          params = {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),

  }

  ##
  # Rewards
  ##

  rewards = {
      # ---------- General ----------
      "is_terminated": RewardTermCfg(
          func = mdp.is_terminated,
          weight = 0.0,
      ),

      # ---------- Root penalties ----------
      "lin_vel_z_l2": RewardTermCfg(
          func=mdp.lin_vel_z_l2,
          weight=0.0,
      ),
      "ang_vel_xy_l2": RewardTermCfg(
          func=mdp.ang_vel_xy_l2,
          weight=0.0,
      ),
      "flat_orientation_l2": RewardTermCfg(
          func=mdp.flat_orientation_l2,
          weight=0.0,
      ),
      "base_height_l2": RewardTermCfg(
          func=mdp.base_height_l2,
          weight=0.0,
          params={
              "asset_cfg": SceneEntityCfg("robot", body_names=()),
              "sensor_name": "height_scanner_base",
              "target_height": 0.0,
          },
      ),
      "body_lin_acc_l2": RewardTermCfg(
          func=mdp.body_lin_acc_l2,
          weight=0.0,
          params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
      ),

      # ---------- Joint penalties ----------
      "joint_torques_l2": RewardTermCfg(
          func=mdp.joint_torques_l2,
          weight=0.0,
          params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*"))},
      ),
      "joint_vel_l2": RewardTermCfg(
          func=mdp.joint_vel_l2,
          weight=0.0,
          params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*"))},
      ),
      "joint_acc_l2": RewardTermCfg(
          func=mdp.joint_acc_l2,
          weight=0.0,
          params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*"))},
      ),
      # here is create_joint_deviation_l1_rewterm（joint_deviation_l1）
      "joint_pos_limits": RewardTermCfg(
          func=mdp.joint_pos_limits,
          weight=0.0,
          params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*"))},
      ),
      "joint_vel_limits": RewardTermCfg(
          func=mdp.joint_vel_limits,
          weight=0.0,
          params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*")), "soft_ratio": 1.0},
      ),
      "joint_power": RewardTermCfg(
          func=mdp.joint_power,
          weight=0.0,
          params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*"))},
      ),
      "stand_still_without_cmd": RewardTermCfg(
          func=mdp.stand_still_without_cmd,
          weight=0.0,
          params={
              "command_name": "base_velocity",
              "command_threshold": 0.1,
              "asset_cfg": SceneEntityCfg("robot", joint_names=(".*")),
          },
      ),
      "joint_pos_penalty": RewardTermCfg(
          func=mdp.joint_pos_penalty,
          weight=0.0,
          params={
              "command_name": "base_velocity",
              "asset_cfg": SceneEntityCfg("robot", joint_names=(".*")),
              "stand_still_scale": 5.0,
              "velocity_threshold": 0.5,
              "command_threshold": 0.1,
          },
      ),
      "wheel_vel_penalty": RewardTermCfg(
          func=mdp.wheel_vel_penalty,
          weight=0.0,
          params={
              "asset_cfg": SceneEntityCfg("robot", joint_names=()),
              "sensor_name": "feet_ground_contact",
              "command_name": "base_velocity",
              "velocity_threshold": 0.5,
              "command_threshold": 0.1,
          },
      ),
      "wheel_vel_stand_penalty": RewardTermCfg(
          func=mdp.wheel_vel_stand_penalty,
          weight=0.0,
          params={
              "asset_cfg": SceneEntityCfg("robot", joint_names=()),
              "command_name": "base_velocity",
              "command_threshold": 0.1,
          },
      ),
      "joint_mirror": RewardTermCfg(
          func=mdp.joint_mirror,
          weight=0.0,
          params={
              "asset_cfg": SceneEntityCfg("robot"),
              "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
          },
      ),
      "action_mirror": RewardTermCfg(
          func=mdp.action_mirror,
          weight=0.0,
          params={
              "asset_cfg": SceneEntityCfg("robot"),
              "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
          },
      ),
      "action_sync": RewardTermCfg(
          func=mdp.action_sync,
          weight=0.0,
          params={
              "asset_cfg": SceneEntityCfg("robot"),
              "joint_groups": [
                  ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                  ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                  ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
              ],
          },
      ),

      # ---------- Action penalties ----------
      "applied_torque_limits": RewardTermCfg(
          func=mdp.applied_torque_limits,
          weight=0.0,
          params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*"))},
      ),
      "action_rate_l2": RewardTermCfg(
          func=mdp.action_rate_l2,
          weight=0.0,
      ),

      # ---------- Contact sensors ----------
      "undesired_contacts": RewardTermCfg(
          func=mdp.undesired_contacts,
          weight=0.0,
          params={
              "sensor_name": "nonfeet_all_contact",
              "threshold": 1.0,
          },
      ),
      "contact_forces": RewardTermCfg(
          func=mdp.contact_forces,
          weight=0.0,
          params={
            "sensor_name": "feet_ground_contact", 
            "threshold": 100.0},
      ),

      # ---------- Velocity-tracking rewards ----------
      "track_lin_vel_xy_exp": RewardTermCfg(
          func=mdp.track_lin_vel_xy_exp,
          weight=0.0,
          params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
      ),
      "track_ang_vel_z_exp": RewardTermCfg(
          func=mdp.track_ang_vel_z_exp,
          weight=0.0,
          params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
      ),

      # ---------- Others ----------
      "feet_air_time": RewardTermCfg(
          func=mdp.feet_air_time,
          weight=0.0,
          params={
              "command_name": "base_velocity",
              "mode_time": 0.3,
              "velocity_threshold": 0.5,
              "command_threshold": 0.1,
              "asset_cfg": SceneEntityCfg("robot"),
              "sensor_name": "feet_ground_contact",
          },
      ),
      "feet_gait": RewardTermCfg(
          func=mdp.feet_gait,  
          weight=0.0,
          params={
              "period":0.6,
              "offset" : [0.0, 0.5],
              "threshold": 0.56,
              "command_threshold": 0.1,
              "command_name": "base_velocity",
              "sensor_name": "feet_ground_contact",
          },
      ),
      "feet_contact": RewardTermCfg(
          func=mdp.feet_contact,
          weight=0.0,
          params={
              "sensor_name": "feet_ground_contact",
              "command_name": "base_velocity",
              "expect_contact_num": 2,
          },
      ),
      "feet_contact_without_cmd": RewardTermCfg(
          func=mdp.feet_contact_without_cmd,
          weight=0.0,
          params={
              "sensor_name": "feet_ground_contact",
              "command_name": "base_velocity",
          },
      ),
      "feet_stumble": RewardTermCfg(
          func=mdp.feet_stumble,
          weight=0.0,
          params={"sensor_name": "feet_ground_contact"},
      ),
      "feet_slide": RewardTermCfg(
          func=mdp.feet_slide,
          weight=0.0,
          params={
              "sensor_name": "feet_ground_contact",
              "asset_cfg": SceneEntityCfg("robot"),
          },
      ),
      "feet_height": RewardTermCfg(
          func=mdp.feet_height,
          weight=0.0,
          params={
              "asset_cfg": SceneEntityCfg("robot", body_names=()),
              "tanh_mult": 2.0,
              "target_height": 0.05,
              "command_name": "base_velocity",
          },
      ),
      "feet_height_body": RewardTermCfg(
          func=mdp.feet_height_body,
          weight=0.0,
          params={
              "asset_cfg": SceneEntityCfg("robot", body_names=()),
              "tanh_mult": 2.0,
              "target_height": -0.3,
              "command_name": "base_velocity",
          },
      ),
      "feet_distance_y_exp": RewardTermCfg(
          func=mdp.feet_distance_y_exp,
          weight=0.0,
          params={
              "std": math.sqrt(0.25),
              "asset_cfg": SceneEntityCfg("robot", body_names=()),
              "stance_width": float,  
          },
      ),
      # "feet_distance_xy_exp": {...},  

      "upward": RewardTermCfg(
          func=mdp.upward,
          weight=0.0,
      ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # "fell_over": TerminationTermCfg(
    #   func=mdp.bad_orientation,
    #   params={"limit_angle": math.radians(70.0)},
    # ),
    "terrain_out_of_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
      time_out=True,
    ),
    # not used in robot_lab
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "nonfeet_ground_contact"},
      time_out=True,
    ),
    ## The observation group 'actor' returned by the environment contains NaN values.
    # This cuases the rsl-rl training to crash.
    "nan_term": TerminationTermCfg(
      func=mdp.nan_detection,
      params={},
      time_out=False,
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      # params={"command_name": "base_velocity"},
    ),
    # "command_vel": CurriculumTermCfg(
    #   func=mdp.commands_vel,
    #   params={
    #     "command_name": "base_velocity",
    #     "velocity_stages": [
    #       {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    #       {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
    #       {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
    #     ],
    #   },
    # ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(

    # --- Physics ---

    decimation=4,
    sim=SimulationCfg(
      ## when play, this below appears(where the policy is bad, many part of the dog contact with ground)
      # broadphase overflow - please increase nconmax to 38 or naconmax to 38
      # broadphase overflow - please increase nconmax to 37 or naconmax to 37
      nconmax=50,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scanner, height_scanner_base),
      num_envs=1,
      extent=2.0,
    ),

    # --- Eposode ---

    episode_length_s=20.0,
    is_finite_horizon=False,
    scale_rewards_by_dt=True,

    # --- Manager ---

    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,

    # --- Misc ---

    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    seed = None,
  )
