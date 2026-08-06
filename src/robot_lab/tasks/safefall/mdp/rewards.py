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
