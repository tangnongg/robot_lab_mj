"""Unitree Go2W velocity environment configurations."""

import math
from typing import Literal

from robot_lab.asset_zoo.robots import (
  get_unitree_go2w_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from robot_lab.tasks.go2w import mdp
from robot_lab.tasks.go2w.mdp import UniformThresholdVelocityCommandCfg
from robot_lab.tasks.go2w.velocity_env_cfg import make_velocity_env_cfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

def unitree_go2w_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2W rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # --- Physics ---

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_unitree_go2w_robot_cfg()}

  # Set raycast sensor frame to Go2w base.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scanner":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      # sensor.frame.name = "base"
      sensor.frame.name = "imu_site"

  # Wire foot height scan to per-foot sites.
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scanner":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
  
  # feet_ground_cfg = ContactSensorCfg(
  #   name="feet_ground_contact",
  #   primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
  #   secondary=ContactMatch(mode="body", pattern="terrain"),
  #   fields=("found", "force"),
  #   reduce="netforce",
  #   num_slots=1, 
  #   track_air_time=True,
  #   history_length=4,
  # )

  foot_body_names = ('FL_foot',  'FR_foot', 'RL_foot', 'RR_foot')
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="body", pattern=foot_body_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1, 
    track_air_time=True,
    history_length=4,
  )

  # used in illegal_contact, but not used in robot_lab
  nonfeet_ground_cfg = ContactSensorCfg(
    name="nonfeet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=r".*_collision\d*$", entity="robot", exclude=tuple(geom_names)),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  target_body_names = ('base', 
                       'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 
                       'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 
                       'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot', 
                       'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot')
  exclude_body_names = ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')
  
  nonfeet_all_cfg = ContactSensorCfg(
    name="nonfeet_all_contact",
    # when mode="body", Available string:
    # 'base', 'Head_upper', 'Head_lower', 'imu', 'terrain_scan', 'height_scanner_base', 
    # 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_calflower', 'FL_calflower1', 'FL_foot_motor', 'FL_foot', 
    # 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_calflower', 'FR_calflower1', 'FR_foot_motor', 'FR_foot', 
    # 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_calflower', 'RL_calflower1', 'RL_foot_motor', 'RL_foot', 
    # 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_calflower', 'RR_calflower1', 'RR_foot_motor', 'RR_foot'
    primary=ContactMatch(mode="body", pattern=target_body_names, entity="robot", exclude=exclude_body_names ),
    secondary_policy='any',
    # secondary=ContactMatch(mode="body", pattern=("terrain",) + target_body_names, entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfeet_ground_cfg,
    nonfeet_all_cfg
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # --- Eposode ---

  # --- Manager ---

  base_link_name = "base"
  foot_link_name = ".*_foot"
  wheel_joint_name = ".*_foot_joint"
  joint_name = ".*_joint"
  joint_names = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
  )
  # 广度顺序 like robot_lab
  # joint_names = (
  #   # "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  #   # "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  #   # "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
  #   # "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  #   # "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
  # )

  ## Observations
  
  cfg.observations["actor"].terms["joint_pos"].func = mdp.joint_pos_rel_without_wheel
  cfg.observations["actor"].terms["joint_pos"].params["wheel_asset_cfg"] = SceneEntityCfg(
    "robot", 
    joint_names = wheel_joint_name
  )
  cfg.observations["critic"].terms["joint_pos"].func = mdp.joint_pos_rel_without_wheel
  cfg.observations["critic"].terms["joint_pos"].params["wheel_asset_cfg"] = SceneEntityCfg(
    "robot", 
    joint_names = wheel_joint_name
  )
  cfg.observations["actor"].terms["base_lin_vel"].scale = 2.0
  cfg.observations["actor"].terms["base_ang_vel"].scale = 0.25
  cfg.observations["actor"].terms["joint_pos"].scale = 1.0
  cfg.observations["actor"].terms["joint_vel"].scale = 0.05
  del cfg.observations["actor"].terms["base_lin_vel"]
  del cfg.observations["actor"].terms["height_scan"]
  # KeyError: 'asset_cfg'
  # cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"].joint_names = joint_names
  # cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"].joint_names = joint_names
  cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", 
    joint_names = joint_names
  )
  cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", 
    joint_names = joint_names
  )

  ## 
  del cfg.events["randomize_rigid_body_mass_and_inertia"]
  del cfg.events["randomize_com_positions"]
  del cfg.events["randomize_apply_external_force_torque"]
  del cfg.events["randomize_reset_joints"]
  del cfg.events["randomize_joint_stiffness"]
  del cfg.events["randomize_joint_damping"]
  del cfg.events["randomize_reset_base"]
  del cfg.events["randomize_push_robot"]

  ## Actions

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # joint_pos_action.scale = GO2W_ACTION_SCALE # the GO2W_ACTION_SCALE is not caculated in the way of Beyondmimic  
  joint_pos_action.scale = 0.25
  joint_pos_action.use_default_offset = True
  joint_pos_action.clip = None
  joint_pos_action.preserve_order = True

  cfg.actions["joint_vel"] = JointVelocityActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=5,  
    use_default_offset=True,
    clip=None,
    preserve_order=True,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # Available list comes from joint_names variable defined above, but no foot joints?
  joint_pos_action.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25} 
  cfg.actions["joint_pos"].clip = {".*": (-100.0, 100.0)}
  joint_pos_action.actuator_names = joint_names[:-4] # speicify the joint order of joint_pos action

  cfg.actions["joint_vel"].scale = 5.0
  cfg.actions["joint_vel"].clip = {".*": (-100.0, 100.0)}
  cfg.actions["joint_vel"].actuator_names = joint_names[-4:]  # speicify the joint order of joint_vel action

  ## Rewards

  cfg.rewards["joint_vel_wheel_l2"] = RewardTermCfg(
    func=mdp.joint_vel_l2,
    weight=-0.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  cfg.rewards["joint_acc_wheel_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-0.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  cfg.rewards["joint_torques_wheel_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-0.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  cfg.rewards["stand_still_wheel"] = RewardTermCfg(
    func=mdp.stand_still_wheel,
    weight=-0.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_threshold": 0.1,
      "command_name": "base_velocity",
    },
  )

  # General
  cfg.rewards["is_terminated"].weight = 0

  # Root penalties
  cfg.rewards["lin_vel_z_l2"].weight = -2.0
  cfg.rewards["ang_vel_xy_l2"].weight = -0.05

  cfg.rewards["flat_orientation_l2"].weight = 0
  cfg.rewards["base_height_l2"].weight = 0
  cfg.rewards["base_height_l2"].params["target_height"] = 0.40
  cfg.rewards["base_height_l2"].params["asset_cfg"].body_names = base_link_name
  cfg.rewards["body_lin_acc_l2"].weight = 0
  cfg.rewards["body_lin_acc_l2"].params["asset_cfg"].body_names = base_link_name

  # Joint penaltie
  cfg.rewards["joint_torques_l2"].weight = -2.5e-5
  cfg.rewards["joint_torques_l2"].params["asset_cfg"].joint_names = f"^(?!{wheel_joint_name}).*"
  cfg.rewards["joint_torques_wheel_l2"].weight = 0
  cfg.rewards["joint_torques_wheel_l2"].params["asset_cfg"].joint_names = wheel_joint_name
  cfg.rewards["joint_vel_l2"].weight = 0
  cfg.rewards["joint_vel_l2"].params["asset_cfg"].joint_names = f"^(?!{wheel_joint_name}).*"
  cfg.rewards["joint_vel_wheel_l2"].weight = 0
  cfg.rewards["joint_vel_wheel_l2"].params["asset_cfg"].joint_names = wheel_joint_name
  cfg.rewards["joint_acc_l2"].weight = -2.5e-7
  cfg.rewards["joint_acc_l2"].params["asset_cfg"].joint_names = f"^(?!{wheel_joint_name}).*"
  cfg.rewards["joint_acc_wheel_l2"].weight = -2.5e-9
  cfg.rewards["joint_acc_wheel_l2"].params["asset_cfg"].joint_names = wheel_joint_name
  # cfg.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
  cfg.rewards["joint_pos_limits"].weight = -5.0
  cfg.rewards["joint_pos_limits"].params["asset_cfg"].joint_names = f"^(?!{wheel_joint_name}).*"
  cfg.rewards["joint_vel_limits"].weight = 0
  cfg.rewards["joint_vel_limits"].params["asset_cfg"].joint_names = wheel_joint_name
  cfg.rewards["joint_power"].weight = -2e-5
  cfg.rewards["joint_power"].params["asset_cfg"].joint_names = f"^(?!{wheel_joint_name}).*"
  cfg.rewards["stand_still_without_cmd"].weight = -2.0
  cfg.rewards["stand_still_without_cmd"].params["asset_cfg"].joint_names = f"^(?!{wheel_joint_name}).*"
  # need to delete, otherwise it causes RuntimeError: Expected all 
  #tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
  # del cfg.rewards["stand_still_wheel"] 
  # cfg.rewards.stand_still_wheel.weight = -0.01
  # cfg.rewards.stand_still_wheel.params["asset_cfg"].joint_names = [cfg.wheel_joint_name]
  cfg.rewards["joint_pos_penalty"].weight = -1.0  # TODO
  cfg.rewards["joint_pos_penalty"].params["asset_cfg"].joint_names = f"^(?!{wheel_joint_name}).*"
  cfg.rewards["wheel_vel_penalty"].weight = 0
  cfg.rewards["wheel_vel_penalty"].params["asset_cfg"].joint_names = wheel_joint_name
  cfg.rewards["wheel_vel_stand_penalty"].weight = -0.05
  cfg.rewards["wheel_vel_stand_penalty"].params["asset_cfg"].joint_names = wheel_joint_name
  cfg.rewards["joint_mirror"].weight = -0.05
  cfg.rewards["joint_mirror"].params["mirror_joints"] = [
      ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
      ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
  ]

  # Action penalties
  cfg.rewards["action_rate_l2"].weight = -0.01

  # Contact sensor
  cfg.rewards["undesired_contacts"].weight = -1.0
  cfg.rewards["contact_forces"].weight = -1.5e-4

  # Velocity-tracking rewards
  cfg.rewards["track_lin_vel_xy_exp"].weight = 3.0
  cfg.rewards["track_ang_vel_z_exp"].weight = 1.5

  # Others
  cfg.rewards["feet_air_time"].weight = 0
  cfg.rewards["feet_contact"].weight = 0
  cfg.rewards["feet_contact_without_cmd"].weight = 0.1
  cfg.rewards["feet_stumble"].weight = 0
  cfg.rewards["feet_slide"].weight = 0
  cfg.rewards["feet_slide"].params["asset_cfg"].body_names = foot_link_name
  cfg.rewards["feet_height"].weight = 0
  cfg.rewards["feet_height"].params["target_height"] = 0.1
  cfg.rewards["feet_height"].params["asset_cfg"].body_names = foot_link_name
  cfg.rewards["feet_height_body"].weight = 0
  cfg.rewards["feet_height_body"].params["target_height"] = -0.2
  cfg.rewards["feet_height_body"].params["asset_cfg"].body_names = foot_link_name
  cfg.rewards["feet_gait"].weight = 0
  cfg.rewards["feet_gait"].params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
  cfg.rewards["upward"].weight = 1.0  

  # del rewards whose weiget is zero
  del cfg.rewards["stand_still_wheel"]
  del cfg.rewards["action_mirror"]
  del cfg.rewards["action_sync"]
  del cfg.rewards["applied_torque_limits"]
  del cfg.rewards["base_height_l2"]
  del cfg.rewards["body_lin_acc_l2"]
  del cfg.rewards["feet_air_time"]
  del cfg.rewards["feet_contact"]
  del cfg.rewards["feet_distance_y_exp"]
  del cfg.rewards["feet_gait"]
  del cfg.rewards["feet_height"]
  del cfg.rewards["feet_height_body"]
  del cfg.rewards["feet_slide"]
  del cfg.rewards["feet_stumble"]
  del cfg.rewards["flat_orientation_l2"]
  del cfg.rewards["is_terminated"]
  del cfg.rewards["joint_torques_wheel_l2"]
  del cfg.rewards["joint_vel_l2"]
  del cfg.rewards["joint_vel_limits"]
  del cfg.rewards["joint_vel_wheel_l2"]
  del cfg.rewards["wheel_vel_penalty"]

  # delete regularization rewards
  del cfg.rewards["lin_vel_z_l2"]
  del cfg.rewards["ang_vel_xy_l2"]
  del cfg.rewards["joint_torques_l2"]
  del cfg.rewards["joint_acc_l2"]
  del cfg.rewards["joint_pos_limits"]
  del cfg.rewards["joint_power"]
  del cfg.rewards["stand_still_without_cmd"]
  del cfg.rewards["joint_pos_penalty"]
  del cfg.rewards["wheel_vel_stand_penalty"]
  del cfg.rewards["joint_mirror"]
  # del cfg.rewards["action_rate_l2"]
  del cfg.rewards["undesired_contacts"]
  del cfg.rewards["contact_forces"]
  # del cfg.rewards["track_lin_vel_xy_exp"]
  # del cfg.rewards["track_ang_vel_z_exp"]
  del cfg.rewards["feet_contact_without_cmd"]
  del cfg.rewards["upward"]
  del cfg.rewards["joint_acc_wheel_l2"]

  ## Terminations

  del cfg.terminations["illegal_contact"]

  ## Commands

  # cfg.commands["base_velocity"].ranges.lin_vel_x = (-1.5, 1.5)
  # cfg.commands["base_velocity"].ranges.lin_vel_y = (-1.5, 1.5)
  # cfg.commands["base_velocity"].ranges.ang_vel_z = (-1.0, 1.0)

  ### --- Misc ---

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    # if args_cli.keyboard:
    #   cfg.scene.num_envs = 1
    #   cfg.terminations.time_out = None
    #   cfg.commands.base_velocity.debug_vis = False
    #   config = Se2KeyboardCfg(
    #       v_x_sensitivity=cfg.commands.base_velocity.ranges.lin_vel_x[1],
    #       v_y_sensitivity=cfg.commands.base_velocity.ranges.lin_vel_y[1],
    #       omega_z_sensitivity=cfg.commands.base_velocity.ranges.ang_vel_z[1],
    #   )
    #   controller = Se2Keyboard(config)
    #   cfg.observations.policy.velocity_commands = ObsTerm(
    #       func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
    #   )

    # TODO: add a fixed velocity command generator for play, fixed velocity command
    # from mjlab.managers.command_manager import FixedVelocityCommandCfg
    # cfg.commands["base_velocity"] = FixedVelocityCommandCfg(
    #   ...,
    # )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


# def unitree_go2w_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
#   """Create Unitree Go1 flat terrain velocity configuration."""
#   cfg = unitree_go2w_rough_env_cfg(play=play)

#   cfg.sim.njmax = 300
#   cfg.sim.mujoco.ccd_iterations = 50
#   cfg.sim.contact_sensor_maxmatch = 64
#   cfg.sim.nconmax = None

#   # Switch to flat terrain.
#   assert cfg.scene.terrain is not None
#   cfg.scene.terrain.terrain_type = "plane"
#   cfg.scene.terrain.terrain_generator = None

#   # Remove raycast sensors and collision sensors not needed on flat.
#   remove_sensors = {
#     "terrain_scan",
#   }
#   cfg.scene.sensors = tuple(
#     s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
#   )
#   del cfg.observations["actor"].terms["height_scan"]
#   del cfg.observations["critic"].terms["height_scan"]
#   cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

#   # Remove granular collision rewards (not useful on flat ground).
#   for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
#     cfg.rewards.pop(key, None)

#   # On flat terrain fell_over is sufficient; thigh contact implies fallen.
#   cfg.terminations.pop("illegal_contact", None)
#   cfg.terminations.pop("terrain_out_of_bounds", None)
#   cfg.terminations["fell_over"] = TerminationTermCfg(
#     func=mdp.bad_orientation,
#     params={"limit_angle": math.radians(70.0)},
#   )

#   # Disable terrain curriculum (not present in play mode since rough clears all).
#   cfg.curriculum.pop("terrain_levels", None)

#   if play:
#     base_velocity_cmd = cfg.commands["base_velocity"]
#     assert isinstance(base_velocity_cmd, UniformThresholdVelocityCommandCfg)
#     base_velocity_cmd.ranges.lin_vel_x = (-1.5, 2.0)
#     base_velocity_cmd.ranges.ang_vel_z = (-0.7, 0.7)

#   return cfg
