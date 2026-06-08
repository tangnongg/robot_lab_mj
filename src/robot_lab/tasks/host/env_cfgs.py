"""G1 HoST stand-up environment configurations.

Five variants — ground, platform, wall, slope, prone — each with train and
play modes.  All share the same core structure; variants differ only in
initial pose, reward weights, phase thresholds, and traction-force settings.
"""

from __future__ import annotations

import numpy as np
import torch
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

from mjlab.entity import EntityCfg

from robot_lab.asset_zoo.robots.unitree_g1.g1_constants import (
    get_g1_robot_cfg,
)

from . import mdp


# ---------------------------------------------------------------------------
# Helper — robot scene entity config for all joints
# ---------------------------------------------------------------------------

_JOINT_ASSET_CFG = SceneEntityCfg("robot", joint_names=(".*_joint",))


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


def _build_actor_obs() -> dict[str, ObservationTermCfg]:
    """Actor observation terms matching the original Isaac Lab policy group."""
    return {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            scale=0.25,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-50.0, 50.0),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-10.0, 10.0),
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": _JOINT_ASSET_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-20.0, 20.0),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": _JOINT_ASSET_CFG},
            scale=0.05,
            noise=Unoise(n_min=-0.075, n_max=0.075),
            clip=(-50.0, 50.0),
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            clip=(-50.0, 50.0),
        ),
        "action_rescale": ObservationTermCfg(
            func=mdp.action_rescale_obs,
            clip=(-5.0, 5.0),
        ),
    }


def _build_critic_obs() -> dict[str, ObservationTermCfg]:
    """Critic observation = actor terms + extra privileged information."""
    return {
        **_build_actor_obs(),
        # Full projected gravity (not just the policy-scoped subset)
        "projected_gravity_full": ObservationTermCfg(func=envs_mdp.projected_gravity),
    }


# ---------------------------------------------------------------------------
# Rewards builder
# ---------------------------------------------------------------------------


def _build_rewards(
    task_weight: float = 20.0,
    phase1_height: float = 0.45,
    phase3_height: float = 0.65,
    target_height: float = 0.75,
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
            params={"target_head_height": 1.0, "target_head_margin": 1.0},
        ),
        # ---- Regularization ----
        "regu_dof_acc": RewardTermCfg(
            func=mdp.reward_dof_acc, weight=-2.5e-7
        ),
        "regu_action_rate": RewardTermCfg(
            func=mdp.reward_action_rate, weight=-0.01
        ),
        "regu_smoothness": RewardTermCfg(
            func=mdp.reward_smoothness, weight=-0.01
        ),
        "regu_torques": RewardTermCfg(
            func=mdp.reward_torques, weight=-2.5e-6
        ),
        "regu_joint_power": RewardTermCfg(
            func=mdp.reward_joint_power, weight=-2.5e-5
        ),
        "regu_dof_vel": RewardTermCfg(
            func=mdp.reward_dof_vel, weight=-1e-3
        ),
        "regu_joint_tracking_error": RewardTermCfg(
            func=mdp.reward_joint_tracking_error, weight=-0.00025
        ),
        "regu_dof_pos_limits": RewardTermCfg(
            func=mdp.reward_dof_pos_limits, weight=-5.0
        ),
        "regu_dof_vel_limits": RewardTermCfg(
            func=mdp.reward_dof_vel_limits, weight=-1.0
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
            func=mdp.reward_ground_parallel, weight=20.0
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
        "target_feet_height_var": RewardTermCfg(
            func=mdp.reward_feet_height_var,
            weight=2.5,
            params={"phase3_height": phase3_height},
        ),
        "target_upper_dof_pos": RewardTermCfg(
            func=mdp.reward_target_upper_dof_pos,
            weight=10.0,
            params={"phase3_height": phase3_height},
        ),
        "target_orientation": RewardTermCfg(
            func=mdp.reward_target_orientation,
            weight=10.0,
            params={"phase3_height": phase3_height},
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
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.9, 1.1),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": _JOINT_ASSET_CFG,
            },
        ),
        "reset_buffers": EventTermCfg(
            func=mdp.reset_last_last_action,
            mode="reset",
        ),
        # Interval — traction force (every step)
        "traction_force": EventTermCfg(
            func=mdp.apply_traction_force,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "initial_force": 100.0,
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
        articulation=base_robot_cfg.articulation,
        collisions=base_robot_cfg.collisions,
        sort_actuators=base_robot_cfg.sort_actuators,
    )
    if init_joint_pos is not None:
        robot_cfg.init_state.joint_pos.update(init_joint_pos)

    feet_ground = _make_feet_contact_sensor()
    nonfeet_ground = _make_nonfeet_contact_sensor()

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
            num_envs=1,
            extent=2.5,
        ),
        episode_length_s=10.0,
        is_finite_horizon=False,
        scale_rewards_by_dt=True,
        observations={
            "actor": ObservationGroupCfg(
                terms=_build_actor_obs(),
                concatenate_terms=True,
                enable_corruption=True,
                history_length=6,
            ),
            "critic": ObservationGroupCfg(
                terms=_build_critic_obs(),
                concatenate_terms=True,
                enable_corruption=False,
                history_length=6,
            ),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=1.0,
                use_default_offset=True,
                clip={r".*": (-100.0, 100.0)},
            )
        },
        commands={},  # No commands for stand-up task
        events=_build_events(
            unactuated_steps=unactuated_steps,
            no_orientation=no_orientation,
        ),
        rewards=_build_rewards(
            task_weight=task_weight,
            phase1_height=phase1_height,
            phase3_height=phase3_height,
            target_height=target_height,
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
                    "initial_force": 100.0,
                    "force_decrement": 20.0,
                    "action_rescale_decrement": 0.02,
                    "min_action_rescale": 0.25,
                    "threshold_height": 0.9,
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
    """Ground start: crouching on flat ground, pitched forward."""
    return unitree_g1_host_env_cfg(
        play=play,
        init_pos=(0.0, 0.0, 0.5),
        init_quat=(1.0, 0.0, -1.0, 0.0),  # pitched forward
    )


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
