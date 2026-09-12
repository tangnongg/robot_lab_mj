"""G1 HoST stand-up environment configurations.

Five variants — ground, platform, wall, slope, prone — each with train and
play modes.  All share the same core structure; variants differ only in
initial pose, reward weights, phase thresholds, and traction-force settings.

Matches the official HoST implementation:
- Relative joint position control (target = current_pos + action)
- Actions & observations zeroed during unactuated period
- PD stiffness matching official legged_gym config
- Joint reset range 0.5–1.5 × default
"""

from __future__ import annotations

import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from robot_lab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTUATOR_4010,
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_WAIST,
    get_g1_robot_cfg,
)

from . import mdp


# ---------------------------------------------------------------------------
# Official-PD-gain actuator overrides
# ---------------------------------------------------------------------------
# The mjlab G1 actuators are derived from electric motor parameters (armature,
# reflected inertia, natural frequency), which yields physically realistic but
# lower PD gains than the official HoST implementation.  We override stiffness
# and damping to match the official legged_gym config while keeping the
# original armature and effort_limit.
#
# Official mapping (by substring match):
#   hip     → 150 N·m/rad,   4 N·m·s/rad
#   knee    → 200 N·m/rad,   6 N·m·s/rad
#   ankle   →  40 N·m/rad,   2 N·m·s/rad
#   shoulder→ 100 N·m/rad,   4 N·m·s/rad
#   elbow   → 100 N·m/rad,   4 N·m·s/rad
#   waist   → 100 N·m/rad,   4 N·m·s/rad
#   wrist   → 100 N·m/rad,   4 N·m·s/rad


def _override_stiffness_damping(
    base: BuiltinPositionActuatorCfg,
    stiffness: float,
    damping: float,
) -> BuiltinPositionActuatorCfg:
    """Return a copy of *base* with overridden stiffness and damping."""
    return BuiltinPositionActuatorCfg(
        target_names_expr=base.target_names_expr,
        transmission_type=base.transmission_type,
        armature=base.armature,
        frictionloss=base.frictionloss,
        viscous_damping=base.viscous_damping,
        delay_min_lag=base.delay_min_lag,
        delay_max_lag=base.delay_max_lag,
        delay_hold_prob=base.delay_hold_prob,
        delay_update_period=base.delay_update_period,
        delay_per_env_phase=base.delay_per_env_phase,
        stiffness=stiffness,
        damping=damping,
        effort_limit=base.effort_limit,
    )


# Split actuator groups to match the official per-joint-group PD gains.
# G1_ACTUATOR_7520_14 → "hip" (hip_pitch/hip_yaw) vs "waist" (waist_yaw)
_G1_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
    stiffness=150.0,
    damping=4.0,
    armature=G1_ACTUATOR_7520_14.armature,
    effort_limit=G1_ACTUATOR_7520_14.effort_limit,
)
_G1_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    stiffness=100.0,
    damping=4.0,
    armature=G1_ACTUATOR_7520_14.armature,
    effort_limit=G1_ACTUATOR_7520_14.effort_limit,
)
# G1_ACTUATOR_7520_22 → "hip" (hip_roll) vs "knee"
_G1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint",),
    stiffness=150.0,
    damping=4.0,
    armature=G1_ACTUATOR_7520_22.armature,
    effort_limit=G1_ACTUATOR_7520_22.effort_limit,
)
_G1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    stiffness=200.0,
    damping=6.0,
    armature=G1_ACTUATOR_7520_22.armature,
    effort_limit=G1_ACTUATOR_7520_22.effort_limit,
)
# G1_ACTUATOR_5020 → all become 100/4 (shoulder, elbow, wrist_roll)
_G1_ACTUATOR_UPPER = _override_stiffness_damping(G1_ACTUATOR_5020, 100.0, 4.0)
# G1_ACTUATOR_4010 → wrist_pitch/yaw → 100/4
_G1_ACTUATOR_WRIST = _override_stiffness_damping(G1_ACTUATOR_4010, 100.0, 4.0)
# G1_ACTUATOR_WAIST → waist_pitch/roll → 100/4
_G1_ACTUATOR_WAIST_PITCH_ROLL = _override_stiffness_damping(G1_ACTUATOR_WAIST, 100.0, 4.0)
# G1_ACTUATOR_ANKLE → 40/2
_G1_ACTUATOR_ANKLE_OFFICIAL = _override_stiffness_damping(G1_ACTUATOR_ANKLE, 40.0, 2.0)


_G1_ARTICULATION_OFFICIAL = EntityArticulationInfoCfg(
    actuators=(
        _G1_ACTUATOR_UPPER,
        _G1_ACTUATOR_HIP,
        _G1_ACTUATOR_WAIST_YAW,
        _G1_ACTUATOR_HIP_ROLL,
        _G1_ACTUATOR_KNEE,
        _G1_ACTUATOR_WRIST,
        _G1_ACTUATOR_WAIST_PITCH_ROLL,
        _G1_ACTUATOR_ANKLE_OFFICIAL,
    ),
    soft_joint_pos_limit_factor=0.9,
)

_POLICY_JOINT_NAMES = (
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_yaw_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
)
_JOINT_ASSET_CFG = SceneEntityCfg("robot", joint_names=_POLICY_JOINT_NAMES)


# ---------------------------------------------------------------------------
# Foot / non-foot contact sensor helpers
# ---------------------------------------------------------------------------

_FOOT_GEOM_NAMES = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def _make_feet_contact_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=_FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=4,
    )


def _make_nonfeet_contact_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="nonfeet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r".*_collision\d*$",
            entity="robot",
            exclude=_FOOT_GEOM_NAMES,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=4,
    )


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------


def _build_actor_obs(
    initial_action_rescale: float = 1.0,
) -> dict[str, ObservationTermCfg]:
    """Actor observation terms matching the official HoST policy group.

    Uses custom wrappers that zero observations during the unactuated period
    (first *unactuated_steps* env steps).
    """
    # Noise and scale are handled inside the wrapper functions
    # (so they are applied BEFORE the unactuated zero-gating, matching
    # the official HoST order).  Do NOT set noise/scale on the cfg.
    return {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            clip=(-50.0, 50.0),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            clip=(-10.0, 10.0),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos,
            params={"asset_cfg": _JOINT_ASSET_CFG},
            clip=(-20.0, 20.0),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel,
            params={"asset_cfg": _JOINT_ASSET_CFG},
            clip=(-50.0, 50.0),
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
            clip=(-50.0, 50.0),
        ),
        "action_rescale": ObservationTermCfg(
            func=mdp.action_rescale_obs,
            params={"initial_rescale": initial_action_rescale},
            clip=(-5.0, 5.0),
        ),
    }


def _build_critic_obs(
    initial_action_rescale: float = 1.0,
) -> dict[str, ObservationTermCfg]:
    """Critic observation = actor terms + full projected gravity (privileged)."""
    return {
        **_build_actor_obs(initial_action_rescale),
        "projected_gravity_full": ObservationTermCfg(func=mdp.projected_gravity),
    }


def _build_critic_phase_obs() -> dict[str, ObservationTermCfg]:
    """Privileged phase bit used only to select the value head.

    Keep this one-dimensional input stable so a known-good two-critic
    checkpoint can be resumed while the actor is refined for quiet standing.
    Hold progress is already present in the reward and does not need to be
    exposed as a critic shortcut.
    """
    return {"phase": ObservationTermCfg(func=mdp.critic_phase)}


# ---------------------------------------------------------------------------
# Rewards builder
# ---------------------------------------------------------------------------


def _build_rewards(
    task_weight: float = 20.0,
    phase1_height: float = 0.45,
    phase3_height: float = 0.60,
    target_height: float = 0.75,
    no_orientation: bool = False,
) -> dict[str, RewardTermCfg]:
    """All 27 reward terms, with variant-tunable parameters."""
    return {
        # ---- Task ----
        "task_orientation": RewardTermCfg(
            func=mdp.reward_orientation,
            weight=task_weight,
            params={"phase1_height": phase1_height, "orientation_threshold": 0.99},
        ),
        "task_head_height": RewardTermCfg(
            func=mdp.reward_head_height,
            weight=task_weight,
            # Keep the original broad height shaping: it gives a useful
            # gradient throughout the large supine-to-upright correction and
            # does not make the post-task pose itself an absolute target.
            params={"target_head_height": 1.0, "target_head_margin": 1.0},
        ),
        "task_hold_progress": RewardTermCfg(
            func=mdp.reward_standup_hold_progress,
            # Continuous hold is the post-task objective; keep its progress
            # signal comparable to the two stand-up shaping terms after the
            # environment dt scaling.
            weight=20.0,
            params={
                "threshold_height": 0.60,
                "min_torso_height": 0.66,
                "upright_gravity_z": -0.70,
                # Filled per episode by the outer curriculum: early levels
                # can score a short hold immediately, then the settling delay
                # grows to the final 35 policy steps.
                "settle_steps": 0,
                "quiet_root_ang_vel": 1.0,
                "quiet_root_lin_vel": 0.6,
                "quiet_joint_vel": 2.0,
                "quiet_root_ang_vel_initial": 8.0,
                "quiet_root_lin_vel_initial": 4.0,
                "quiet_joint_vel_initial": 20.0,
                "no_orientation": no_orientation,
            },
        ),
        # ---- Regularization ----
        "regu_dof_acc": RewardTermCfg(
            func=mdp.reward_dof_acc, weight=-2.5e-8
        ),
        "regu_action_rate": RewardTermCfg(
            func=mdp.reward_action_rate, weight=-0.001
        ),
        "regu_smoothness": RewardTermCfg(
            func=mdp.reward_smoothness, weight=-0.001
        ),
        "regu_torques": RewardTermCfg(
            func=mdp.reward_torques, weight=-2.5e-7
        ),
        "regu_joint_power": RewardTermCfg(
            func=mdp.reward_joint_power, weight=-2.5e-6
        ),
        "regu_dof_vel": RewardTermCfg(
            func=mdp.reward_dof_vel, weight=-1e-5
        ),
        "regu_joint_tracking_error": RewardTermCfg(
            func=mdp.reward_joint_tracking_error, weight=-0.025
        ),
        "regu_dof_pos_limits": RewardTermCfg(
            func=mdp.reward_dof_pos_limits, weight=-10.0
        ),
        "regu_dof_vel_limits": RewardTermCfg(
            func=mdp.reward_dof_vel_limits, weight=-0.1
        ),
        # ---- Style ----
        "style_waist_deviation": RewardTermCfg(
            func=mdp.reward_waist_deviation, weight=-10.0
        ),
        "style_hip_yaw_deviation": RewardTermCfg(
            func=mdp.reward_hip_yaw_deviation, weight=-10.0
        ),
        "style_hip_roll_deviation": RewardTermCfg(
            func=mdp.reward_hip_roll_deviation, weight=-10.0
        ),
        "style_shoulder_roll_deviation": RewardTermCfg(
            func=mdp.reward_shoulder_roll_deviation, weight=-2.5
        ),
        "style_left_foot_displacement": RewardTermCfg(
            func=mdp.reward_left_foot_displacement, weight=2.5
        ),
        "style_right_foot_displacement": RewardTermCfg(
            func=mdp.reward_right_foot_displacement, weight=2.5
        ),
        "style_knee_deviation": RewardTermCfg(
            func=mdp.reward_knee_deviation, weight=-0.25
        ),
        "style_shank_orientation": RewardTermCfg(
            func=mdp.reward_shank_orientation, weight=10.0
        ),
        "style_ground_parallel": RewardTermCfg(
            # The MJCF has no ankle keypoint sites. Using one body per ankle
            # makes the variance identically zero, so this term must stay off.
            func=mdp.reward_ground_parallel, weight=0.0
        ),
        "style_feet_distance": RewardTermCfg(
            func=mdp.reward_feet_distance, weight=-10.0
        ),
        "style_ang_vel_xy": RewardTermCfg(
            func=mdp.reward_style_ang_vel_xy, weight=1.0
        ),
        # ---- Post-task (target) ----
        "target_ang_vel_xy": RewardTermCfg(
            func=mdp.reward_target_ang_vel_xy,
            weight=10.0,
            params={"phase3_height": phase3_height},
        ),
        "target_lin_vel_xy": RewardTermCfg(
            func=mdp.reward_target_lin_vel_xy,
            weight=10.0,
            params={"phase3_height": phase3_height},
        ),
        # Relative uprightness is yaw-invariant: projected gravity contains
        # roll/pitch information only, so this does not select a world heading.
        "target_upright_relative": RewardTermCfg(
            func=mdp.reward_target_orientation,
            weight=10.0,
            params={"phase3_height": phase3_height},
        ),
        # ---- Post-task quiet standing ----
        # These costs are gated by the relative upright/height condition. They
        # do not prescribe a world-frame yaw or a hand-authored terminal pose.
        "post_root_ang_vel_l2": RewardTermCfg(
            func=mdp.reward_post_root_ang_vel_l2,
            weight=-4.0,
            params={"phase3_height": phase3_height},
        ),
        "post_root_lin_vel_l2": RewardTermCfg(
            func=mdp.reward_post_root_lin_vel_l2,
            weight=-2.0,
            params={"phase3_height": phase3_height},
        ),
        "post_joint_vel_l2": RewardTermCfg(
            func=mdp.reward_post_joint_vel_l2,
            weight=-0.05,
            params={"phase3_height": phase3_height},
        ),
        "post_joint_acc_l2": RewardTermCfg(
            func=mdp.reward_post_joint_acc_l2,
            weight=-1.0e-6,
            params={"phase3_height": phase3_height},
        ),
        "post_action_rate_l2": RewardTermCfg(
            func=mdp.reward_post_action_rate_l2,
            weight=-0.1,
            params={"phase3_height": phase3_height},
        ),
        "post_smoothness_l2": RewardTermCfg(
            func=mdp.reward_post_smoothness_l2,
            weight=-0.05,
            params={"phase3_height": phase3_height},
        ),
        "post_feet_slip": RewardTermCfg(
            func=mdp.reward_post_feet_slip,
            weight=-5.0,
            params={"phase3_height": phase3_height},
        ),
        "post_pose_drift": RewardTermCfg(
            func=mdp.reward_post_pose_drift,
            weight=-1.0,
        ),
        "post_action_drift": RewardTermCfg(
            func=mdp.reward_post_action_drift,
            weight=-0.5,
        ),
        # One joint-space term covers the complete mirrored standing pose:
        # waist yaw, both legs, and every policy-controlled arm/hand joint.
        "post_symmetric_pose": RewardTermCfg(
            func=mdp.reward_post_symmetric_pose,
            weight=-1.5,
            params={"phase3_height": phase3_height, "settle_steps": 35},
        ),
        # The only separate geometric term is for the support polygon: feet
        # stay compact, aligned front/back, and parallel in the root frame.
        "post_foot_alignment": RewardTermCfg(
            func=mdp.reward_post_foot_alignment,
            weight=-1.0,
            params={"phase3_height": phase3_height, "settle_steps": 35},
        ),
        "target_base_height": RewardTermCfg(
            func=mdp.reward_target_base_height,
            weight=10.0,
            params={"phase3_height": phase3_height, "target_height": target_height},
        ),
    }


# ---------------------------------------------------------------------------
# Events builder
# ---------------------------------------------------------------------------


def _build_events(
    unactuated_steps: int = 30,
    no_orientation: bool = False,
    initial_force: float = 200.0,
    initial_action_rescale: float = 1.0,
    initial_hold_steps: int = 5,
) -> dict[str, EventTermCfg]:
    return {
        # Reset
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_scaled,
            mode="reset",
            params={
                "scale_range": (0.9, 1.1),
                "offset_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": _JOINT_ASSET_CFG,
            },
        ),
        "reset_buffers": EventTermCfg(
            func=mdp.reset_last_last_action,
            mode="reset",
        ),
        "reset_action_rescale": EventTermCfg(
            func=mdp.reset_action_rescale,
            mode="reset",
            params={"initial_rescale": initial_action_rescale},
        ),
        "reset_standup_success": EventTermCfg(
            func=mdp.reset_standup_success,
            mode="reset",
            params={
                "initial_hold_steps": initial_hold_steps,
            },
        ),
        # Interval — traction force (every step)
        "traction_force": EventTermCfg(
            func=mdp.apply_traction_force,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "initial_force": initial_force,
                "no_orientation": no_orientation,
                "unactuated_steps": unactuated_steps,
            },
        ),
        # Interval — update last-last-action buffer for smoothness (every step)
        "update_last_last_action": EventTermCfg(
            func=mdp.update_last_last_action,
            mode="interval",
            interval_range_s=(0.0, 0.0),
        ),
        "update_standup_success": EventTermCfg(
            func=mdp.update_standup_success,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                # The previous latch was above the height reached by the
                # emerging upright policy.  These thresholds still exclude
                # kneeling while allowing the post critic to observe and
                # stabilize the learned standing state.
                # Broad candidate latch: the hold gate below remains the
                # actual standing requirement and continues to demand the
                # higher torso/root height.
                "threshold_height": 0.55,
                "upright_gravity_z": -0.55,
                "min_torso_height": 0.45,
                # Enter the post critic after one upright frame.  The same
                # outer curriculum then grows the hold and quiet requirements.
                "candidate_steps": 1,
                "hold_steps": initial_hold_steps,
                "hold_threshold_height": 0.60,
                "hold_min_torso_height": 0.56,
                "hold_upright_gravity_z": -0.70,
                "hold_threshold_height_initial": 0.52,
                "hold_min_torso_height_initial": 0.45,
                "hold_upright_gravity_z_initial": -0.55,
                "hold_settle_steps": 0,
                "quiet_root_ang_vel": 1.0,
                "quiet_root_lin_vel": 0.6,
                "quiet_joint_vel": 2.0,
                "quiet_root_ang_vel_initial": 8.0,
                "quiet_root_lin_vel_initial": 4.0,
                "quiet_joint_vel_initial": 20.0,
                "no_orientation": no_orientation,
            },
        ),
    }


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def unitree_g1_host_env_cfg(
    play: bool = False,
    *,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.5),
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, -1.0, 0.0),
    init_joint_pos: dict[str, float] | None = None,
    task_weight: float = 20.0,
    phase1_height: float = 0.45,
    phase3_height: float = 0.65,
    target_height: float = 0.75,
    unactuated_steps: int = 30,
    no_orientation: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build a HoST stand-up environment config for the Unitree G1.

    Args:
        play: If True, produce a play-mode config (fewer envs, no noise).
        init_pos: Initial base position (x, y, z) in world frame.
        init_quat: Initial base orientation quaternion (w, x, y, z).
        init_joint_pos: Optional per-joint position overrides.
        task_weight: Weight for the task-achievement reward terms.
        phase1_height: Base height threshold for early-standing gate.
        phase3_height: Base height threshold for post-task target rewards.
        target_height: Desired standing base height.
        unactuated_steps: Number of steps before traction force activates.
        no_orientation: If True, skip orientation gating for traction force.
    """
    # Build robot config with variant-specific init state.
    # get_g1_robot_cfg() returns a fresh EntityCfg each time BUT the
    # module-level KNEES_BENT_KEYFRAME is shared as init_state.  Clone it
    # so per-variant overrides don't leak.
    base_robot_cfg = get_g1_robot_cfg()
    robot_cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_quat,
            joint_pos=dict(base_robot_cfg.init_state.joint_pos),
            joint_vel=dict(base_robot_cfg.init_state.joint_vel),
        ),
        spec_fn=base_robot_cfg.spec_fn,
        articulation=_G1_ARTICULATION_OFFICIAL,
        collisions=base_robot_cfg.collisions,
        sort_actuators=base_robot_cfg.sort_actuators,
    )
    if init_joint_pos is not None:
        robot_cfg.init_state.joint_pos.update(init_joint_pos)

    feet_ground = _make_feet_contact_sensor()
    nonfeet_ground = _make_nonfeet_contact_sensor()
    initial_force = 0.0 if play else 200.0
    initial_action_rescale = 0.25 if play else 1.0
    initial_hold_steps = 1
    final_hold_steps = 250

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
            entities={"robot": robot_cfg},
            sensors=(feet_ground, nonfeet_ground),
            num_envs=2048,
            extent=2.5,
        ),
        episode_length_s=10.0,
        is_finite_horizon=False,
        scale_rewards_by_dt=True,
        observations={
            "actor": ObservationGroupCfg(
                terms=_build_actor_obs(initial_action_rescale),
                concatenate_terms=True,
                enable_corruption=True,
                history_length=6,
            ),
            "critic": ObservationGroupCfg(
                terms=_build_critic_obs(initial_action_rescale),
                concatenate_terms=True,
                enable_corruption=False,
                history_length=6,
            ),
            "critic_phase": ObservationGroupCfg(
                terms=_build_critic_phase_obs(),
                concatenate_terms=True,
                enable_corruption=False,
                history_length=1,
            ),
        },
        actions={
            "joint_pos": mdp.RelativeJointPositionActionCfg(
                entity_name="robot",
                actuator_names=_POLICY_JOINT_NAMES,
                scale=1.0,
                unactuated_steps=unactuated_steps,
                initial_rescale=initial_action_rescale,
                clip={r".*": (-100.0, 100.0)},
            )
        },
        commands={},  # No commands for stand-up task
        events=_build_events(
            unactuated_steps=unactuated_steps,
            no_orientation=no_orientation,
            initial_force=initial_force,
            initial_action_rescale=initial_action_rescale,
            initial_hold_steps=initial_hold_steps,
        ),
        rewards=_build_rewards(
            task_weight=task_weight,
            phase1_height=phase1_height,
            phase3_height=phase3_height,
            target_height=target_height,
            no_orientation=no_orientation,
        ),
        terminations={
            "time_out": TerminationTermCfg(
                func=envs_mdp.time_out, time_out=True
            ),
            "joint_vel_exceeded": TerminationTermCfg(
                func=mdp.joint_velocity_exceeded,
                params={"threshold": 300.0},
            ),
            "base_vel_exceeded": TerminationTermCfg(
                func=mdp.base_velocity_exceeded,
                params={"threshold": 20.0},
            ),
        },
        curriculum={
            "traction_force": CurriculumTermCfg(
                func=mdp.traction_force_curriculum,
                params={
                    "initial_force": initial_force,
                    # A single shared level maps linearly to force, action
                    # scale, and hold duration. It changes only after a full
                    # performance window, avoiding per-environment drift.
                    "initial_action_rescale": initial_action_rescale,
                    "min_action_rescale": 0.25,
                    "initial_hold_steps": initial_hold_steps,
                    "final_hold_steps": final_hold_steps,
                    "initial_hold_settle_steps": 0,
                    "final_hold_settle_steps": 35,
                    # A 1024-episode window gives each difficulty level
                    # enough samples while allowing the full assistance and
                    # quiet-motion course to complete in one practical run.
                    "window_episodes": 1024,
                    "promote_success_rate": 0.50,
                    "demote_success_rate": 0.15,
                    "promote_level_step": 0.10,
                    "demote_level_step": 0.05,
                    "quiet_root_ang_vel_initial": 8.0,
                    "quiet_root_ang_vel_final": 1.0,
                    "quiet_root_lin_vel_initial": 4.0,
                    "quiet_root_lin_vel_final": 0.6,
                    "quiet_joint_vel_initial": 20.0,
                    "quiet_joint_vel_final": 2.0,
                },
            ),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="pelvis",
            distance=2.0,
            elevation=-10.0,
            azimuth=90.0,
        ),
        seed=None,
    )

    if play:
        cfg.scene.num_envs = 50
        cfg.episode_length_s = 30.0
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["critic"].enable_corruption = False

    return cfg


# ---------------------------------------------------------------------------
# Per-variant convenience factories
# ---------------------------------------------------------------------------


def unitree_g1_ground_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Ground start: lying face-up (supine) on flat ground."""
    return unitree_g1_host_env_cfg(
        play=play,
        init_pos=(0.0, 0.0, 0.5),
        init_quat=(0.70710678, 0.0, -0.70710678, 0.0),
    )


def unitree_g1_ground_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Backward-compatible alias for the complete end-to-end task."""
    return unitree_g1_ground_env_cfg(play=play)


def unitree_g1_ground_hold_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Backward-compatible alias for the complete end-to-end task."""
    return unitree_g1_ground_env_cfg(play=play)


def unitree_g1_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Platform start: on a raised platform, pitched backward."""
    return unitree_g1_host_env_cfg(
        play=play,
        init_pos=(0.0, 0.0, 0.5),
        init_quat=(1.0, 0.0, 1.0, 0.0),  # pitched backward
        task_weight=25.0,
    )


def unitree_g1_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Wall start: upright against a wall, legs extended."""
    return unitree_g1_host_env_cfg(
        play=play,
        init_pos=(0.0, 0.0, 0.45),
        init_quat=(1.0, 0.0, 0.0, 0.0),  # upright
        init_joint_pos={
            "left_hip_pitch_joint": -1.5,
            "right_hip_pitch_joint": -1.5,
            "left_knee_joint": 0.0,
            "right_knee_joint": 0.0,
        },
        task_weight=25.0,
        unactuated_steps=50,
    )


def unitree_g1_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Slope start: on a slope, pitched backward, knees bent."""
    return unitree_g1_host_env_cfg(
        play=play,
        init_pos=(0.0, 0.0, 0.8),
        init_quat=(1.0, 0.0, 1.0, 0.0),  # pitched backward
        init_joint_pos={
            "left_knee_joint": 1.0,
            "right_knee_joint": 1.0,
            "left_elbow_joint": 0.0,
            "right_elbow_joint": 0.0,
        },
        task_weight=25.0,
        phase1_height=0.4,
        phase3_height=0.6,
        target_height=0.70,
    )


def unitree_g1_prone_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Prone start: lying face-down on the ground."""
    return unitree_g1_host_env_cfg(
        play=play,
        init_pos=(0.0, 0.0, 0.5),
        init_quat=(1.0, 0.0, -1.0, 0.0),  # pitched forward 180°
        no_orientation=True,
    )
