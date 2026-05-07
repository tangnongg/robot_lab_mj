"""Unitree Go2W constants."""

from pathlib import Path

import mujoco

from robot_lab import ROBOT_LAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2W_XML: Path = (
  ROBOT_LAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2w" / "xmls" / "go2w.xml"
)
assert GO2W_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2W_XML))


##
# Actuator config.
##

GO2W_ACTUATOR_LEGS = DcMotorActuatorCfg(
  target_names_expr=(
    ".*hip_.*", ".*thigh_.*", ".*calf_.*",
  ),
  saturation_effort=23.5,
  velocity_limit=30.0,
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=None,
)

GO2W_ACTUATOR_WHEELS = DcMotorActuatorCfg(
  target_names_expr=(".*_foot_joint",),
  effort_limit=23.5,
  saturation_effort=23.5,
  velocity_limit=30.0,
  stiffness=0.0,
  damping=0.5,
  armature=None,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.45),
  joint_pos={
    ".*L_hip_joint": 0.0,
    ".*R_hip_joint": -0.0,
    "F.*_thigh_joint": 0.8,
    "R.*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
    ".*_foot_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions.
# Foot collisions are given custom condim, friction.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  # Harden all collision geoms.
  solref=(0.01, 1),
  # Configure feet colliders. Other colliders are frictionless (condim=1).
  condim={_foot_regex: 6, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

GO1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2W_ACTUATOR_LEGS,
    GO2W_ACTUATOR_WHEELS,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_unitree_go2w_robot_cfg() -> EntityCfg:
  """Get a fresh Go1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO1_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_unitree_go2w_robot_cfg())

  viewer.launch(robot.spec.compile())
