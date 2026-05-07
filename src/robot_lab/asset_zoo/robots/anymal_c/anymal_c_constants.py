"""ANYbotics ANYmal C constants."""

from pathlib import Path

import mujoco

from robot_lab import ROBOT_LAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
# from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ANYMAL_C_XML: Path = ROBOT_LAB_SRC_PATH / "asset_zoo" / "robots" / "anymal_c" / "xmls" / "anymal_c.xml"
assert ANYMAL_C_XML.exists()

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ANYMAL_C_XML))
  return spec

##
# Actuator config.
##

EFFORT_LIMIT = 80.0

# Random small armature since we don't know the real value.
ARMATURE = 0.005

# PD gains derived from armature, targeting 10 Hz natural frequency.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

ANYMAL_C_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*HAA", ".*HFE", ".*KFE"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.54),
  joint_pos={
    ".*HAA": 0.0,
    "LF_HFE": 0.4,
    "RF_HFE": 0.4,
    "LH_HFE": -0.4,
    "RH_HFE": -0.4,
    "LF_KFE": -0.8,
    "RF_KFE": -0.8,
    "LH_KFE": 0.8,
    "RH_KFE": 0.8,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[LR][FH]_foot$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", _foot_regex),
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp={_foot_regex: (0.015, 1, 0.03)},
)

##
# Final config.
##

ANYMAL_C_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ANYMAL_C_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_anymal_c_robot_cfg() -> EntityCfg:
  """Get a fresh ANYmal C robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ANYMAL_C_ARTICULATION,
  )


ANYMAL_C_ACTION_SCALE: dict[str, float] = {}
for _a in ANYMAL_C_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _d = _a.damping
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    ANYMAL_C_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_anymal_c_robot_cfg())

  viewer.launch(robot.spec.compile())
