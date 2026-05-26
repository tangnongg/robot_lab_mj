"""Unitree B2 constants."""

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

B2_XML: Path = (
  ROBOT_LAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_b2" / "xmls" / "b2.xml"
)
assert B2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(B2_XML))


##
# Actuator config.
##

B2_ACTUATOR_HIP_THIGH = DcMotorActuatorCfg(
  target_names_expr=(
    ".*hip_.*", ".*thigh_.*",
  ),
  effort_limit=200.0,
  saturation_effort=200.0,
  velocity_limit=23.0,
  stiffness=160.0,
  damping=5.0,
  frictionloss=0.0,
  armature=None,
)

B2_ACTUATOR_CALF = DcMotorActuatorCfg(
  target_names_expr=(".*calf_.*",),
  effort_limit=320.0,
  saturation_effort=320.0,
  velocity_limit=14.0,
  stiffness=160.0,
  damping=5.0,
  frictionloss=0.0,
  armature=None,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.58),
  joint_pos={
    ".*L_hip_joint": 0.0, # hip mirror
    ".*R_hip_joint": -0.0,
    ".*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
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

B2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    B2_ACTUATOR_HIP_THIGH,
    B2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_unitree_b2_robot_cfg() -> EntityCfg:
  """Get a fresh B2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=B2_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_unitree_b2_robot_cfg())

  viewer.launch(robot.spec.compile())
