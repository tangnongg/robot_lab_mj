"""Custom reward functions for SafeFall damage mitigation task.

Implements the damage-aware reward from the SafeFall paper (Meng et al., 2025):

  r_total = r_impact + r_regulation

  r_impact = w_c * r_contact + w_j * r_joint + w_e * r_torque

Impact and regularization functions return non-negative costs and use negative
weights. The post-fall settling function is the exception: it returns a
bounded positive score after terrain contact, paired with a residual-motion
cost so the final body state is quiet.

Migrated from IsaacLab to mjlab. Key API changes:
- asset.data.applied_torque → asset.data.qfrc_actuator
- asset.data.root_pos_w → asset.data.root_link_pos_w
- asset.data.root_lin_vel_w → asset.data.root_link_lin_vel_w
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .metrics import ImpactLoadAccumulator
from .terminations import simulation_state_invalid

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _stable_cost(
    env: ManagerBasedRlEnv,
    cost: torch.Tensor,
    max_cost: float,
) -> torch.Tensor:
    """Bound non-physical tails and clear costs for states being reset."""
    stable = ~simulation_state_invalid(env)
    return torch.nan_to_num(cost, nan=0.0, posinf=max_cost, neginf=0.0).clamp(
        min=0.0, max=max_cost
    ) * stable

# ---------------------------------------------------------------------------
# Component heterogeneity weights (paper §III-D, Eq.3)
# geom name → vulnerability weight w_s ∈ {1000, 1, 0.5}
# ---------------------------------------------------------------------------
# High:   head, hands                                 → w_s = 1000
# Medium: shanks, shoulders, feet                      → w_s = 1
# Low:    torso, thighs, elbows, pelvis, hips, etc.   → w_s = 0.5
# ---------------------------------------------------------------------------
_VULNERABILITY_HIGH = 1000.0
_VULNERABILITY_MED = 1.0
_VULNERABILITY_LOW = 0.5

_GEOM_VULNERABILITY: dict[str, float] = {
    "head_collision": _VULNERABILITY_HIGH,
    "left_hand_collision": _VULNERABILITY_HIGH,
    "right_hand_collision": _VULNERABILITY_HIGH,
    # medium
    "left_shin_collision": _VULNERABILITY_MED,
    "right_shin_collision": _VULNERABILITY_MED,
    "left_shoulder_yaw_collision": _VULNERABILITY_MED,
    "right_shoulder_yaw_collision": _VULNERABILITY_MED,
    **{
        f"{side}_foot{i}_collision": _VULNERABILITY_MED
        for side in ("left", "right")
        for i in range(1, 8)
    },
    # low
    "torso_collision": _VULNERABILITY_LOW,
    "pelvis_collision": _VULNERABILITY_LOW,
    "left_thigh_collision": _VULNERABILITY_LOW,
    "right_thigh_collision": _VULNERABILITY_LOW,
    "left_hip_collision": _VULNERABILITY_LOW,
    "right_hip_collision": _VULNERABILITY_LOW,
    "left_elbow_yaw_collision": _VULNERABILITY_LOW,
    "right_elbow_yaw_collision": _VULNERABILITY_LOW,
    "left_wrist_collision": _VULNERABILITY_LOW,
    "right_wrist_collision": _VULNERABILITY_LOW,
    "left_linkage_brace_collision": _VULNERABILITY_LOW,
    "right_linkage_brace_collision": _VULNERABILITY_LOW,
}

# ---------------------------------------------------------------------------
# Paper r_contact (Eq. 3): component‑heterogeneity contact force penalty
#   r_contact = (1/N) Σ I{c_i}·w_{s,i}·[f_{contact,i} − m_i·g]_+²
#             + α · max_i { I{c_i}·w_{s,i}·[f_{contact,i} − m_i·g]_+² }
# where N = Σ I{c_i} (active contacts) and α = 0.3
# ---------------------------------------------------------------------------


class ContactForcePenalty(ManagerTermBase):
    """Per‑link contact force penalty with component heterogeneity weights.

    Reads per‑geom contact forces from a ContactSensor and applies
    vulnerability weights w_s ∈ {1000, 1, 0.5} per the paper (head/hands
    → 1000, shanks/shoulders → 1, torso/thighs/elbows → 0.5).

    MuJoCo's parent filtering excludes adjacent-link collisions; Stage II also
    includes non-adjacent self contacts.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._alpha = float(cfg.params.get("alpha", 0.3))
        self._sensor = env.scene[cfg.params.get("sensor_name", "body_ground_contact")]
        asset: Entity = env.scene["robot"]
        mj_model = env.sim.mj_model

        local_geom_ids, resolved_names = asset.find_geoms(
            self._sensor.primary_names, preserve_order=True
        )
        if resolved_names != self._sensor.primary_names:
            raise RuntimeError("Contact sensor and entity geom order do not match")
        geom_ids = [
            int(asset.indexing.geom_ids[local_id]) for local_id in local_geom_ids
        ]
        geom_body_ids = [int(mj_model.geom_bodyid[geom_id]) for geom_id in geom_ids]

        component_names: list[str] = []
        component_indices: dict[str, int] = {}
        geom_group_ids: list[int] = []
        for name in self._sensor.primary_names:
            component_name = re.sub(r"_foot[1-7]_collision$", "_foot_collision", name)
            if component_name not in component_indices:
                component_indices[component_name] = len(component_names)
                component_names.append(component_name)
            geom_group_ids.append(component_indices[component_name])
        self._geom_group_ids = torch.tensor(
            geom_group_ids, device=env.device, dtype=torch.long
        )

        component_body_ids = [-1] * len(component_names)
        component_vulnerability = [_VULNERABILITY_LOW] * len(component_names)
        for geom_idx, group_idx in enumerate(geom_group_ids):
            body_id = geom_body_ids[geom_idx]
            if component_body_ids[group_idx] not in (-1, body_id):
                raise RuntimeError("A contact component spans multiple MuJoCo bodies")
            component_body_ids[group_idx] = body_id
            component_vulnerability[group_idx] = max(
                component_vulnerability[group_idx],
                _GEOM_VULNERABILITY.get(
                    self._sensor.primary_names[geom_idx], _VULNERABILITY_LOW
                ),
            )

        body_counts: dict[int, int] = {}
        for body_id in component_body_ids:
            body_counts[body_id] = body_counts.get(body_id, 0) + 1

        # A body can expose multiple semantic collision regions (notably torso
        # and head). Split its gravitational loading instead of subtracting the
        # complete body weight once per geom.
        masses = [
            float(mj_model.body_mass[body_id]) / body_counts[body_id]
            for body_id in component_body_ids
        ]
        self._gravity_load = torch.tensor(
            masses, device=env.device, dtype=torch.float
        ) * abs(float(mj_model.opt.gravity[2]))
        self._vulnerability = torch.tensor(
            [
                value for value in component_vulnerability
            ],
            device=env.device,
            dtype=torch.float,
        )

    def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
        data = self._sensor.data
        if data.force is None:
            return torch.zeros(self.num_envs, device=self.device)

        if data.force_history is not None:
            # [B, geom, substep, xyz] -> [B, substep, geom, xyz]
            force_vectors = data.force_history.permute(0, 2, 1, 3)
        else:
            force_vectors = data.force.unsqueeze(1)

        grouped = torch.zeros(
            force_vectors.shape[0],
            force_vectors.shape[1],
            self._gravity_load.numel(),
            3,
            device=self.device,
            dtype=force_vectors.dtype,
        )
        grouped.index_add_(2, self._geom_group_ids, force_vectors)
        force = torch.linalg.vector_norm(grouped, dim=-1)

        contact_mask = force > 1.0e-6
        excess = torch.clamp(force - self._gravity_load.view(1, 1, -1), min=0.0)
        weighted = (
            contact_mask
            * self._vulnerability.view(1, 1, -1)
            * excess.square()
        )
        active = contact_mask.sum(dim=-1).clamp(min=1)
        cost = weighted.sum(dim=-1) / active + self._alpha * weighted.max(dim=-1).values
        return _stable_cost(env, cost.max(dim=1).values, max_cost=5.0e10) * _impact_phase(env)


# ---------------------------------------------------------------------------
# Paper r_torque (Eq. 5):  r_torque = Σ_j [|τ_j| / τ̄_j − 1]_+²
# ---------------------------------------------------------------------------


def _impact_loads(env: ManagerBasedRlEnv) -> ImpactLoadAccumulator:
    try:
        return env._safefall_impact_loads
    except AttributeError as exc:
        raise RuntimeError(
            "SafeFall impact rewards require the ImpactLoadAccumulator metric"
        ) from exc


def reward_joint_torques(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Normalized torque ratio penalty matching paper Eq.5.

    Penalizes joint torques that exceed the actuator's maximum rated torque.
    τ̄_i is the actuator effort_limit per joint.
    """
    return _stable_cost(
        env, _impact_loads(env).consume_torque_cost(), max_cost=1.0e3
    ) * _impact_phase(env)


# ---------------------------------------------------------------------------
# Paper r_joint (Eq. 4):  r_joint = Σ_j ‖f_{joint,j} − f_thresh,j‖²
#
# f_{joint,j} is the joint-reaction force that maintains kinematic
# constraints between adjacent links during impact propagation.
# MuJoCo-Warp exposes cfrc_int, the spatial interaction wrench transmitted
# between each body and its parent. The substep accumulator uses its force
# component and a configurable mechanical threshold.
# ---------------------------------------------------------------------------

def reward_joint_reaction(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Joint reaction force penalty — proxy for paper Eq.4.

    Uses MuJoCo ``cfrc_int`` interaction forces accumulated at 200 Hz by
    :class:`ImpactLoadAccumulator`.
    """
    return _stable_cost(
        env, _impact_loads(env).consume_joint_force_cost(), max_cost=2.5e8
    ) * _impact_phase(env)


# ---------------------------------------------------------------------------
# Regularization rewards (r_regulation)
# ---------------------------------------------------------------------------


def reward_action_rate(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Penalize rapid action changes for smooth protective motions."""
    cost = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1
    )
    # During impact, rapid target changes can be protective.  Once the hold
    # phase begins, the same motion should be strongly damped.
    phase = _hold_phase(env)
    phase_weight = 0.1 + 0.9 * phase
    return _stable_cost(env, cost, max_cost=2.0e4) * phase_weight


def reward_joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive joint velocities."""
    asset: Entity = env.scene[asset_cfg.name]
    cost = torch.sum(torch.square(asset.data.joint_vel), dim=-1)
    return _stable_cost(env, cost, max_cost=1.2e6) * _hold_phase(env)


def reward_joint_acc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Entity = env.scene[asset_cfg.name]
    cost = torch.sum(torch.square(asset.data.joint_acc), dim=-1)
    return _stable_cost(env, cost, max_cost=2.0e7) * _hold_phase(env)


def reward_joint_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions near limits."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    lower = asset.data.soft_joint_pos_limits[..., 0]
    upper = asset.data.soft_joint_pos_limits[..., 1]
    below = torch.clamp(lower - pos, min=0.0)
    above = torch.clamp(pos - upper, min=0.0)
    cost = torch.sum(below + above, dim=-1)
    return _stable_cost(env, cost, max_cost=100.0)


def reward_simulation_failure(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-shot penalty for leaving the simulator's physical validity range."""
    return simulation_state_invalid(env).float()


def _settling_motion(env: ManagerBasedRlEnv):
    try:
        return env._safefall_settling_motion
    except AttributeError as exc:
        raise RuntimeError(
            "SafeFall settling rewards require the SettlingMotionAccumulator metric"
        ) from exc


def _hold_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Training-only phase mask; it is never passed to the actor."""
    return _settling_motion(env).phase.float()


def _impact_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    return 1.0 - _hold_phase(env)


def reward_post_fall_stability(
    env: ManagerBasedRlEnv,
    **_: object,
) -> torch.Tensor:
    """Reward low motion after terrain contact and a sustained quiet interval."""
    return _settling_motion(env).score


def penalty_post_fall_motion(
    env: ManagerBasedRlEnv,
    **_: object,
) -> torch.Tensor:
    """Penalize residual root/joint motion after terrain contact."""
    return _settling_motion(env).motion_cost


def penalty_post_fall_pose_drift(
    env: ManagerBasedRlEnv,
    **_: object,
) -> torch.Tensor:
    """Penalize drifting away from the naturally derived post-impact pose.

    The reference is captured after a short impact-absorption delay, so this
    term does not impose a fixed upright/folded target on the fall.
    """
    return _settling_motion(env).pose_drift


def penalty_post_fall_action_drift(
    env: ManagerBasedRlEnv,
    **_: object,
) -> torch.Tensor:
    """Penalize changing the joint target after the natural pose is locked."""
    return _settling_motion(env).action_drift


# ---------------------------------------------------------------------------
# Post-impact posture safety
# ---------------------------------------------------------------------------


class PostFallHeadClearancePenalty(ManagerTermBase):
    """Keep the head collision sphere clear of the terrain during hold.

    The regular paper contact term is deliberately active only while absorbing
    the impact.  Without a hold-phase term, a policy can therefore obtain a
    quiet but unsafe terminal state by resting its head on the ground.  This
    term uses the actual MuJoCo collision sphere rather than pelvis height, so
    it remains meaningful for side, back, and prone falls alike.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
        local_ids, names = asset.find_geoms(("head_collision",), preserve_order=True)
        if names != ["head_collision"]:
            raise RuntimeError("SafeFall head-clearance reward requires head_collision")
        self._head_geom_id = int(asset.indexing.geom_ids[local_ids[0]])
        self._head_radius = float(env.sim.mj_model.geom_size[self._head_geom_id, 0])
        self._min_clearance = float(cfg.params.get("min_clearance", 0.04))

    def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
        # The plane terrain is at each environment origin.  ``geom_xpos`` is
        # the sphere centre, hence subtract its radius to obtain true clearance.
        terrain_z = env.scene.env_origins[:, 2]
        surface_z = env.sim.data.geom_xpos[:, self._head_geom_id, 2] - self._head_radius
        clearance = surface_z - terrain_z
        cost = torch.square(torch.relu(self._min_clearance - clearance) / self._min_clearance)
        return _stable_cost(env, cost, max_cost=100.0) * _hold_phase(env)


class PostFallHeadContactPenalty(ManagerTermBase):
    """Penalize terrain load borne by the head.

    ``hold_only=False`` creates a weaker peak-impact term, while the default
    creates the much stronger sustained-load term used after pose lock.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        sensor = env.scene[cfg.params.get("sensor_name", "ground_contact")]
        try:
            self._head_index = sensor.primary_names.index("head_collision")
        except ValueError as exc:
            raise RuntimeError("SafeFall ground-contact sensor must include head_collision") from exc
        self._free_force = float(cfg.params.get("free_force", 5.0))
        self._force_scale = float(cfg.params.get("force_scale", 50.0))
        self._hold_only = bool(cfg.params.get("hold_only", True))

    def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
        data = env.scene["ground_contact"].data
        if data.force is None:
            return torch.zeros(env.num_envs, device=env.device)
        # Use every 200 Hz sample represented by this policy step.  A short
        # re-contact should not be hidden by the final net force sample.
        force = data.force_history if data.force_history is not None else data.force.unsqueeze(2)
        head_force = torch.linalg.vector_norm(force[:, self._head_index], dim=-1).amax(dim=-1)
        cost = torch.square(torch.relu(head_force - self._free_force) / self._force_scale)
        phase_mask = _hold_phase(env) if self._hold_only else 1.0
        return _stable_cost(env, cost, max_cost=100.0) * phase_mask


class PostFallExcessLimbHeightPenalty(ManagerTermBase):
    """Discourage a high-energy, suspended-limb terminal pose.

    This is a mass-weighted hinge on the COM height of all left/right limb
    links.  It leaves the final configuration unconstrained below the height
    margin, so it does not encode a hand-designed lying pose; it merely makes
    a curled leg held high against gravity less attractive.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
        local_ids = [
            index for index, name in enumerate(asset.body_names)
            if name.startswith(("left_", "right_"))
        ]
        if not local_ids:
            raise RuntimeError("SafeFall limb-height reward could not resolve limb bodies")
        body_ids = asset.indexing.body_ids[local_ids]
        masses = torch.as_tensor(
            env.sim.mj_model.body_mass[body_ids.cpu().numpy()],
            device=env.device,
            dtype=torch.float,
        )
        self._body_ids = torch.as_tensor(local_ids, device=env.device, dtype=torch.long)
        self._mass_weights = masses / masses.sum().clamp(min=1.0e-6)
        self._height_margin = float(cfg.params.get("height_margin", 0.30))

    def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
        heights = (
            env.scene["robot"].data.body_com_pos_w[:, self._body_ids, 2]
            - env.scene.env_origins[:, None, 2]
        )
        excess = torch.relu(heights - self._height_margin) / self._height_margin
        cost = torch.sum(self._mass_weights * torch.square(excess), dim=-1)
        return _stable_cost(env, cost, max_cost=100.0) * _hold_phase(env)


class ExcessLegFlexionPenalty(ManagerTermBase):
    """Discourage the leg-folding posture that raises the CoM during a fall.

    This is intentionally a *soft excess* penalty rather than a reference-pose
    tracker.  Normal knee bending still provides impact compliance; only knee
    flexion beyond ``knee_limit`` and extreme hip pitch are penalized.  It is
    active in both impact and hold so the impact policy cannot create a highly
    curled configuration that the hold policy would later inherit.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
        knee_ids, knee_names = asset.find_joints(
            ("left_knee_joint", "right_knee_joint"), preserve_order=True
        )
        hip_ids, hip_names = asset.find_joints(
            ("left_hip_pitch_joint", "right_hip_pitch_joint"), preserve_order=True
        )
        if knee_names != ["left_knee_joint", "right_knee_joint"]:
            raise RuntimeError("SafeFall leg-flexion reward requires both knee joints")
        if hip_names != ["left_hip_pitch_joint", "right_hip_pitch_joint"]:
            raise RuntimeError("SafeFall leg-flexion reward requires both hip-pitch joints")
        self._knee_ids = torch.as_tensor(knee_ids, device=env.device, dtype=torch.long)
        self._hip_ids = torch.as_tensor(hip_ids, device=env.device, dtype=torch.long)
        self._knee_limit = float(cfg.params.get("knee_limit", 1.45))
        self._hip_limit = float(cfg.params.get("hip_limit", 1.25))
        self._hip_weight = float(cfg.params.get("hip_weight", 0.5))

    def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
        joint_pos = env.scene["robot"].data.joint_pos
        knee_excess = torch.relu(joint_pos[:, self._knee_ids] - self._knee_limit)
        hip_excess = torch.relu(torch.abs(joint_pos[:, self._hip_ids]) - self._hip_limit)
        cost = (
            torch.mean(torch.square(knee_excess / self._knee_limit), dim=-1)
            + self._hip_weight
            * torch.mean(torch.square(hip_excess / self._hip_limit), dim=-1)
        )
        return _stable_cost(env, cost, max_cost=100.0)


class ImpactLegFoldPenalty(ManagerTermBase):
    """Keep the knees and ankles from folding into the upper body on impact.

    Joint angles alone are not sufficient once the robot is rotating: a leg can
    reach the torso through a coordinated hip/knee motion without violating one
    scalar angle limit.  This term uses body geometry and is active only during
    the impact window.  The final hold pose remains unconstrained except by the
    regular leg-flexion and settling terms.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
        required = (
            "torso_link",
            "left_knee_link",
            "right_knee_link",
            "left_ankle_pitch_link",
            "right_ankle_pitch_link",
        )
        missing = [name for name in required if name not in asset.body_names]
        if missing:
            raise RuntimeError(f"SafeFall impact leg-fold reward missing bodies: {missing}")
        self._torso_id = asset.body_names.index("torso_link")
        self._knee_ids = torch.as_tensor(
            [asset.body_names.index("left_knee_link"), asset.body_names.index("right_knee_link")],
            device=env.device,
            dtype=torch.long,
        )
        self._ankle_ids = torch.as_tensor(
            [
                asset.body_names.index("left_ankle_pitch_link"),
                asset.body_names.index("right_ankle_pitch_link"),
            ],
            device=env.device,
            dtype=torch.long,
        )
        self._knee_min_distance = float(cfg.params.get("knee_min_distance", 0.24))
        self._ankle_min_distance = float(cfg.params.get("ankle_min_distance", 0.20))
        self._distance_scale = float(cfg.params.get("distance_scale", 0.10))
        self._ankle_weight = float(cfg.params.get("ankle_weight", 0.5))

    def __call__(self, env: ManagerBasedRlEnv, **_: object) -> torch.Tensor:
        body_pos = env.scene["robot"].data.body_com_pos_w
        torso = body_pos[:, self._torso_id : self._torso_id + 1]
        knee_dist = torch.linalg.vector_norm(body_pos[:, self._knee_ids] - torso, dim=-1)
        ankle_dist = torch.linalg.vector_norm(body_pos[:, self._ankle_ids] - torso, dim=-1)
        knee_excess = torch.relu(self._knee_min_distance - knee_dist) / self._distance_scale
        ankle_excess = torch.relu(self._ankle_min_distance - ankle_dist) / self._distance_scale
        cost = torch.mean(torch.square(knee_excess), dim=-1) + self._ankle_weight * torch.mean(
            torch.square(ankle_excess), dim=-1
        )
        return _stable_cost(env, cost, max_cost=100.0) * _impact_phase(env)
