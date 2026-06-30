"""SafeFall environment configurations for mjlab.

Implements the SafeFall protective control policy training environment
where the G1 humanoid is initialized in falling states and learns to
minimize damage through controlled protective maneuvers.

Paper (Meng et al., 2025) key design:
- Fixed short episodes: 40 steps = 0.8s (paper Section III-D)
- Damage-aware reward: r_total = r_impact + r_regulation
  r_impact = w_c·r_contact + w_j·r_joint + w_e·r_torque (Eq.1-5)
- Asymmetric actor-critic (paper Section III-D)
- Two-stage curriculum (Stage I implemented; Stage II requires fall predictor)
- Domain randomization (paper Table II)
"""

from __future__ import annotations

import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.dr import actuator as dr_actuator
from mjlab.envs.mdp.dr import body as dr_body
from mjlab.envs.mdp.dr import geom as dr_geom
from mjlab.envs.mdp.dr import joint as dr_joint
from mjlab.envs.mdp.observations import (
    base_ang_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)
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
    FULL_COLLISION,
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
# PD-gain actuator overrides matching IsaacLab SafeFall G1 config
# ---------------------------------------------------------------------------
# hip     → 150 N·m/rad,   5 N·m·s/rad
# knee    → 200 N·m/rad,   6 N·m·s/rad
# ankle   →  40 N·m/rad,   2 N·m·s/rad
# shoulder→ 100 N·m/rad,   4 N·m·s/rad
# elbow   → 100 N·m/rad,   4 N·m·s/rad
# waist   → 100 N·m/rad,   4 N·m·s/rad
# wrist   → 100 N·m/rad,   4 N·m·s/rad


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


# Hip group: split 7520_14 into hip_pitch/hip_yaw (150/5) and waist_yaw (100/4).
_G1_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
    stiffness=150.0,
    damping=5.0,
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
_G1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint",),
    stiffness=150.0,
    damping=5.0,
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
_G1_ACTUATOR_ANKLE_SAFEFALL = _override_stiffness_damping(G1_ACTUATOR_ANKLE, 40.0, 2.0)
_G1_ACTUATOR_UPPER = _override_stiffness_damping(G1_ACTUATOR_5020, 100.0, 4.0)
_G1_ACTUATOR_WRIST = _override_stiffness_damping(G1_ACTUATOR_4010, 100.0, 4.0)
_G1_ACTUATOR_WAIST_PITCH_ROLL = _override_stiffness_damping(G1_ACTUATOR_WAIST, 100.0, 4.0)

_SAFEFALL_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        _G1_ACTUATOR_UPPER,
        _G1_ACTUATOR_HIP,
        _G1_ACTUATOR_WAIST_YAW,
        _G1_ACTUATOR_HIP_ROLL,
        _G1_ACTUATOR_KNEE,
        _G1_ACTUATOR_WRIST,
        _G1_ACTUATOR_WAIST_PITCH_ROLL,
        _G1_ACTUATOR_ANKLE_SAFEFALL,
    ),
    soft_joint_pos_limit_factor=0.9,
)

_JOINT_ASSET_CFG = SceneEntityCfg("robot", joint_names=(".*_joint",))


# ---------------------------------------------------------------------------
# Contact sensor — paper r_contact (Eq.3)
# ---------------------------------------------------------------------------

def _make_contact_sensor() -> ContactSensorCfg:
    """Match all robot non-foot collision geoms against the terrain.

    Uses ``reduce="none"`` so each geom's individual 3D contact force
    is preserved in the output — needed for the component-heterogeneity
    weights in Eq.3.
    """
    return ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r".*(?:hip|thigh|shin|knee|torso|head|shoulder|elbow|wrist|hand|pelvis|linkage_brace).*_collision\d*$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=1,
    )


# ---------------------------------------------------------------------------
# Standing initial pose (matches IsaacLab SafeFall G1 init_state)
# ---------------------------------------------------------------------------
_SAFEFALL_STANDING_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.8),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "left_hip_yaw_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_pitch_joint": -0.3,
        "left_knee_joint": 0.6,
        "left_ankle_pitch_joint": -0.3,
        "left_ankle_roll_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.3,
        "right_knee_joint": 0.6,
        "right_ankle_pitch_joint": -0.3,
        "right_ankle_roll_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.3,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.5,
        "right_shoulder_pitch_joint": 0.3,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.5,
    },
    joint_vel={".*": 0.0},
)


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------

def _build_actor_obs() -> dict[str, ObservationTermCfg]:
    """Deployable observation terms matching paper Section III-D.

    Paper: pelvis orientation (r,p), joint states (q, q̇), previous actions
    a_{t-1}, angular velocity ω, projected gravity g_b, stacked over 5
    timesteps.
    """
    return {
        "pelvis_orientation": ObservationTermCfg(
            func=mdp.pelvis_orientation,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-3.14, 3.14),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=base_ang_vel,
            scale=0.25,
            noise=Unoise(n_min=-0.2 * 0.25, n_max=0.2 * 0.25),
            clip=(-50.0, 50.0),
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-10.0, 10.0),
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": _JOINT_ASSET_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-20.0, 20.0),
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": _JOINT_ASSET_CFG},
            scale=0.05,
            noise=Unoise(n_min=-1.5 * 0.05, n_max=1.5 * 0.05),
            clip=(-50.0, 50.0),
        ),
        "last_action": ObservationTermCfg(
            func=last_action,
            clip=(-50.0, 50.0),
        ),
    }


def _build_critic_obs() -> dict[str, ObservationTermCfg]:
    """Critic observation = actor terms (noise-free) + privileged info.

    Paper Section III-D: "The critic additionally accesses privileged simulation
    state including global root positions, velocities and center-of-mass."
    """
    return {
        # Actor terms (no noise — critic sees ground truth)
        "pelvis_orientation": ObservationTermCfg(func=mdp.pelvis_orientation),
        "base_ang_vel": ObservationTermCfg(func=base_ang_vel, scale=0.25),
        "projected_gravity": ObservationTermCfg(func=projected_gravity),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel, params={"asset_cfg": _JOINT_ASSET_CFG},
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel, params={"asset_cfg": _JOINT_ASSET_CFG}, scale=0.05,
        ),
        "last_action": ObservationTermCfg(func=last_action),
        "base_height": ObservationTermCfg(func=mdp.base_height),
        # Privileged terms (not available on real hardware).
        "root_pos_w": ObservationTermCfg(func=_privileged_root_pos),
        "root_lin_vel_w": ObservationTermCfg(func=_privileged_root_lin_vel),
        "root_com_w": ObservationTermCfg(func=_privileged_root_com),
    }


# ---------------------------------------------------------------------------
# Privileged observation helpers (not importable on real hardware)
# ---------------------------------------------------------------------------

def _privileged_root_pos(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Global root position (privileged)."""
    return env.scene[asset_cfg.name].data.root_link_pos_w


def _privileged_root_lin_vel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Global root linear velocity (privileged)."""
    return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def _privileged_root_com(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Center of mass position in world frame (privileged)."""
    return env.scene[asset_cfg.name].data.root_com_pos_w


# ---------------------------------------------------------------------------
# Rewards — paper damage-aware formulation + proxy terms
# ---------------------------------------------------------------------------


def _build_rewards() -> dict[str, RewardTermCfg]:
    return {
        # -- r_impact (Eq.2): w_c·r_contact + w_j·r_joint + w_e·r_torque --
        # r_contact (Eq.3): ContactSensor not providing force data yet.
        "r_contact": RewardTermCfg(
            func=mdp.reward_contact_force,
            weight=-1.0,  # w_c — placeholder; sensor forces unavailable
        ),
        # r_joint (Eq.4): qfrc_external proxy
        "r_joint": RewardTermCfg(
            func=mdp.reward_joint_reaction,
            weight=-5.0e-7,  # w_j — qfrc_external can reach hundreds of N·m
        ),
        # r_torque (Eq.5): normalized torque ratio
        "r_torque": RewardTermCfg(
            func=mdp.reward_joint_torques,
            weight=-2.5e-6,  # w_e — per‑joint ratio already in [0, 1] range
        ),
        # -- r_regulation --
        "action_rate": RewardTermCfg(
            func=mdp.reward_action_rate,
            weight=-0.01,
        ),
        "joint_vel": RewardTermCfg(
            func=mdp.reward_joint_vel,
            weight=-1e-4,
        ),
        "joint_acc": RewardTermCfg(
            func=mdp.reward_joint_acc,
            weight=-2.5e-7,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.reward_joint_pos_limits,
            weight=-10.0,
        ),
    }


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def _build_terminations() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fall_completed": TerminationTermCfg(
            func=mdp.fall_completed,
            params={"height_threshold": 0.15, "velocity_threshold": 0.3},
        ),
        "joint_vel_exceeded": TerminationTermCfg(
            func=mdp.joint_velocity_exceeded,
            params={"threshold": 200.0},
        ),
    }


# ---------------------------------------------------------------------------
# Events — Stage I/II reset + domain randomization (paper Table II)
# ---------------------------------------------------------------------------


def _build_reset_event(stage2: bool = False) -> EventTermCfg:
    """Return the appropriate reset event for the given curriculum stage."""
    if stage2:
        return EventTermCfg(
            func=mdp.reset_falling_from_bank,
            mode="reset",
            params={
                "bank_path": "models/stage2_state_bank.pt",
                "pos_noise": 0.05,
                "vel_noise": 0.1,
            },
        )
    return EventTermCfg(
        func=mdp.reset_falling_state,
        mode="reset",
        params={
            "height_range": (0.4, 0.9),
            "vel_range": (0.0, 2.0),
            "ang_vel_range": (-1.0, 1.0),
        },
    )


def _build_events(stage2: bool = False) -> dict[str, EventTermCfg]:
    reset_term = _build_reset_event(stage2)
    return {
        # 0 — Reset: falling configurations (Stage I random / Stage II bank)
        "reset_falling": reset_term,
        # ── Startup domain randomization (paper Table II) ──
        # 1 — Friction  U(0.3, 1.0)
        "dr_01_friction": EventTermCfg(
            func=dr_geom.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
                "ranges": (0.3, 1.0),
            },
        ),
        # 2 — Restitution  U(0.0, 0.5)  — TODO
        #     MuJoCo stores restitution in geom‑pair solref/solimp, not on
        #     individual geoms.  mjlab's DR module has pair_friction but no
        #     pair_solref / pair_solimp randomizer.  Needs a custom DR
        #     function that iterates mjModel.pair_solref and applies
        #     U(0.0, 0.5) to the first component.
        # "dr_02_restitution": EventTermCfg(
        #     func=...,
        #     mode="startup",
        #     params={...},
        # ),
        # 3 — Base mass offset  U(−1.0, 3.0) kg (log‑uniform approximation)
        "dr_03_mass": EventTermCfg(
            func=dr_body.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "alpha_range": (-0.1, 0.1),  # mass scale ~ exp(2α)
            },
        ),
        # 4 — Base CoM offset  x,y~U(−0.05,0.05), z~U(−0.01,0.01)
        "dr_04_com_offset": EventTermCfg(
            func=dr_body.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.01, 0.01)},
                "distribution": "uniform",
                "operation": "add",
            },
        ),
        # 5 — Joint stiffness scale  logU(0.7, 1.5)
        # 6 — Joint damping scale    logU(0.5, 3.0)
        "dr_05_06_pd_gains": EventTermCfg(
            func=dr_actuator.pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
                "kp_range": (0.7, 1.5),
                "kd_range": (0.5, 3.0),
                "distribution": "log_uniform",
                "operation": "scale",
            },
        ),
        # 7 — Joint position limits  N(0, 0.02)
        "dr_07_joint_limits": EventTermCfg(
            func=dr_joint.joint_limits,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "ranges": (-0.02, 0.02),
                "distribution": "gaussian",
                "operation": "add",
            },
        ),
    }


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def unitree_g1_safefall_env_cfg(
    play: bool = False,
    stage2: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build a SafeFall environment config for the Unitree G1.

    Paper configuration:
    - Physics: 200 Hz (timestep 0.005), policy: 50 Hz (decimation 4)
    - Episode: 40 steps = 0.8 s (paper Section III-D)
    - Action: target joint positions for PD controller (29 DoF)
    - Asymmetric actor-critic with 5-frame observation history

    Args:
        play: If True, use a play-mode config (fewer envs, no noise).
        stage2: If True, sample initial states from the Stage II bank
            (predictor-flagged falling states) instead of random configs.
    """
    base_robot_cfg = get_g1_robot_cfg()
    # Both stages use full collision geometry so the ContactSensor
    # for r_contact (Eq.3) always works.
    collisions = (FULL_COLLISION,)
    robot_cfg = EntityCfg(
        init_state=_SAFEFALL_STANDING_INIT,
        spec_fn=base_robot_cfg.spec_fn,
        articulation=_SAFEFALL_ARTICULATION,
        collisions=collisions,
        sort_actuators=base_robot_cfg.sort_actuators,
    )

    cfg = ManagerBasedRlEnvCfg(
        decimation=4,                     # 200 Hz physics → 50 Hz policy
        sim=SimulationCfg(
            nconmax=128,
            njmax=1024,
            contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(
                timestep=0.005,           # 200 Hz
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
            sensors=(_make_contact_sensor(),),
            num_envs=4096,
            extent=2.5,
        ),
        episode_length_s=0.8,             # Paper: fixed 40 steps at 50 Hz
        is_finite_horizon=False,
        scale_rewards_by_dt=True,
        observations={
            "actor": ObservationGroupCfg(
                terms=_build_actor_obs(),
                concatenate_terms=True,
                enable_corruption=True,
                history_length=5,         # Paper: 5-frame temporal context
            ),
            "critic": ObservationGroupCfg(
                terms=_build_critic_obs(),
                concatenate_terms=True,
                enable_corruption=False,  # Critic sees clean privileged info
                history_length=5,
            ),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=1.0,
                use_default_offset=True,
            )
        },
        commands={},
        events=_build_events(stage2=stage2),
        rewards=_build_rewards(),
        terminations=_build_terminations(),
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
        cfg.episode_length_s = 5.0       # Longer for visualization
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["critic"].enable_corruption = False

    return cfg
