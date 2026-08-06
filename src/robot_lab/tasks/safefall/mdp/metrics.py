"""Physics-substep load tracking for the SafeFall damage rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg


_RATED_TORQUE_BY_PATTERN = (
    ("hip_roll_joint", 139.0),
    ("knee_joint", 139.0),
    ("hip_pitch_joint", 88.0),
    ("hip_yaw_joint", 88.0),
    ("waist_yaw_joint", 88.0),
    ("ankle_", 50.0),
    ("waist_roll_joint", 50.0),
    ("waist_pitch_joint", 50.0),
    ("wrist_pitch_joint", 5.0),
    ("wrist_yaw_joint", 5.0),
    ("shoulder_", 25.0),
    ("elbow_joint", 25.0),
    ("wrist_roll_joint", 25.0),
)


def rated_joint_torques(joint_names: list[str] | tuple[str, ...]) -> list[float]:
    """Resolve G1 rated torques in the simulator's actual joint order."""
    values: list[float] = []
    for name in joint_names:
        try:
            values.append(next(value for pattern, value in _RATED_TORQUE_BY_PATTERN if pattern in name))
        except StopIteration as exc:
            raise ValueError(f"No SafeFall rated torque configured for joint {name!r}") from exc
    return values


class ImpactLoadAccumulator:
    """Track peak joint force and external joint torque at 200 Hz.

    Reward terms are evaluated once per policy step (50 Hz), while impact peaks can
    last only one physics substep.  The metrics manager calls this term after every
    physics step; the reward terms then consume the accumulated maxima.
    """

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._env = env
        self._asset = env.scene["robot"]
        self._joint_dof_ids = self._asset.data.indexing.joint_v_adr.long()
        mj_model = env.sim.mj_model
        joint_body_ids = [
            int(mj_model.dof_bodyid[int(dof_id)])
            for dof_id in self._joint_dof_ids.cpu().tolist()
        ]
        self._joint_body_ids = torch.tensor(joint_body_ids, device=env.device, dtype=torch.long)
        self._rated_torque = torch.tensor(
            rated_joint_torques(self._asset.joint_names),
            device=env.device,
            dtype=torch.float,
        )
        self._peak_joint_force = torch.zeros(env.num_envs, device=env.device)
        self._peak_torque_cost = torch.zeros(env.num_envs, device=env.device)
        env._safefall_impact_loads = self

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        joint_force_threshold: float = 500.0,
    ) -> torch.Tensor:
        data = env.sim.data

        # MuJoCo spatial vectors are [torque_xyz, force_xyz]. cfrc_int is the
        # interaction wrench transmitted from a body to its parent, i.e. the
        # closest available quantity to the paper's per-joint reaction force.
        reaction_force = torch.linalg.vector_norm(
            data.cfrc_int[:, self._joint_body_ids, 3:6], dim=-1
        )
        force_excess = torch.clamp(reaction_force - joint_force_threshold, min=0.0)
        joint_force_cost = torch.sum(force_excess.square(), dim=-1)

        # qfrc_constraint contains generalized force from contacts and other
        # constraints, excluding commanded motor output. This matches Eq. 5's
        # external joint torque more closely than qfrc_actuator.
        external_torque = torch.abs(data.qfrc_constraint[:, self._joint_dof_ids])
        torque_excess = torch.clamp(external_torque / self._rated_torque - 1.0, min=0.0)
        torque_cost = torch.sum(torque_excess.square(), dim=-1)

        self._peak_joint_force = torch.maximum(self._peak_joint_force, joint_force_cost)
        self._peak_torque_cost = torch.maximum(self._peak_torque_cost, torque_cost)
        return torch.maximum(reaction_force.max(dim=-1).values, external_torque.max(dim=-1).values)

    def consume_joint_force_cost(self) -> torch.Tensor:
        value = self._peak_joint_force.clone()
        self._peak_joint_force.zero_()
        return value

    def consume_torque_cost(self) -> torch.Tensor:
        value = self._peak_torque_cost.clone()
        self._peak_torque_cost.zero_()
        return value

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._peak_joint_force[env_ids] = 0.0
        self._peak_torque_cost[env_ids] = 0.0


class SettlingMotionAccumulator:
    """Detect a quiet post-contact state relative to the current fall motion.

    The pre-impact motion peaks are tracked at physics rate.  A settled state
    requires terrain contact and low root/joint motion for several consecutive
    substeps, which prevents a single hand or foot touch from activating the
    post-fall reward while the robot is still rotating.
    """

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._sensor = env.scene["ground_contact"]
        self._asset = env.scene["robot"]
        self._root_dof_ids = self._asset.data.indexing.free_joint_v_adr.long()
        self._joint_dof_ids = self._asset.data.indexing.joint_v_adr.long()
        device = env.device
        n = env.num_envs
        self._peak_lin = torch.zeros(n, device=device)
        self._peak_ang = torch.zeros(n, device=device)
        self._peak_joint = torch.zeros(n, device=device)
        self._contact_seen = torch.zeros(n, device=device, dtype=torch.bool)
        self._contact_steps = torch.zeros(n, device=device, dtype=torch.long)
        self._stable_steps = torch.zeros(n, device=device, dtype=torch.long)
        self._settled = torch.zeros(n, device=device, dtype=torch.bool)
        # The policy is allowed to absorb the impact first.  After a short,
        # contact-relative delay we snapshot the pose/action that naturally
        # emerged and train the policy to hold that snapshot; no hand-written
        # target posture is introduced.
        self._pose_locked = torch.zeros(n, device=device, dtype=torch.bool)
        self._joint_pose_ref = torch.zeros(
            (n, self._joint_dof_ids.numel()), device=device
        )
        self._action_ref = torch.zeros_like(env.action_manager.action)
        self._pose_drift = torch.zeros(n, device=device)
        self._action_drift = torch.zeros(n, device=device)
        self._score = torch.zeros(n, device=device)
        self._motion_cost = torch.zeros(n, device=device)
        env._safefall_settling_motion = self

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        force_threshold: float = 20.0,
        settle_ratio: float = 0.1,
        lin_floor: float = 0.15,
        ang_floor: float = 0.5,
        joint_floor: float = 0.5,
        settle_substeps: int = 20,
        pose_lock_delay_substeps: int = 30,
    ) -> torch.Tensor:
        data = self._sensor.data
        if data.force is not None:
            contact = torch.linalg.vector_norm(data.force, dim=-1).amax(dim=1)
            contact = contact > force_threshold
        elif data.found is not None:
            contact = data.found.amax(dim=1) > 0
        else:
            contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        qvel = env.sim.data.qvel
        qpos = env.sim.data.qpos
        root_qvel = qvel[:, self._root_dof_ids]
        lin = torch.linalg.vector_norm(root_qvel[:, :3], dim=-1)
        ang = torch.linalg.vector_norm(root_qvel[:, 3:6], dim=-1)
        joint = torch.sqrt(torch.mean(torch.square(qvel[:, self._joint_dof_ids]), dim=-1))
        joint_pos = qpos[:, self._asset.data.indexing.joint_q_adr]

        # Peaks before contact define the fall-motion scale for this episode.
        pre_contact = ~self._contact_seen
        self._peak_lin = torch.where(pre_contact, torch.maximum(self._peak_lin, lin), self._peak_lin)
        self._peak_ang = torch.where(pre_contact, torch.maximum(self._peak_ang, ang), self._peak_ang)
        self._peak_joint = torch.where(pre_contact, torch.maximum(self._peak_joint, joint), self._peak_joint)
        self._contact_seen |= contact
        self._contact_steps = torch.where(
            self._contact_seen,
            self._contact_steps + 1,
            torch.zeros_like(self._contact_steps),
        )

        # Freeze the naturally derived post-impact pose only after the impact
        # absorption window.  This lets the policy use the early contact phase
        # to rotate/brace, while subsequent actions are trained to hold still.
        lock_now = (
            self._contact_seen
            & ~self._pose_locked
            & (self._contact_steps >= pose_lock_delay_substeps)
        )
        self._joint_pose_ref = torch.where(
            lock_now[:, None], joint_pos, self._joint_pose_ref
        )
        self._action_ref = torch.where(
            lock_now[:, None], env.action_manager.action, self._action_ref
        )
        self._pose_locked |= lock_now

        pose_error = joint_pos - self._joint_pose_ref
        self._pose_drift = torch.where(
            self._pose_locked,
            torch.mean(torch.square(pose_error), dim=-1),
            torch.zeros_like(self._pose_drift),
        )
        self._action_drift = torch.where(
            self._pose_locked,
            torch.mean(torch.square(env.action_manager.action - self._action_ref), dim=-1),
            torch.zeros_like(self._action_drift),
        )

        lin_limit = torch.maximum(self._peak_lin * settle_ratio, torch.as_tensor(lin_floor, device=env.device))
        ang_limit = torch.maximum(self._peak_ang * settle_ratio, torch.as_tensor(ang_floor, device=env.device))
        joint_limit = torch.maximum(self._peak_joint * settle_ratio, torch.as_tensor(joint_floor, device=env.device))
        quiet = (
            self._contact_seen
            & (lin <= lin_limit)
            & (ang <= ang_limit)
            & (joint <= joint_limit)
        )
        self._stable_steps = torch.where(
            quiet, self._stable_steps + 1, torch.zeros_like(self._stable_steps)
        )
        self._settled = self._stable_steps >= settle_substeps

        lin_ratio = lin / lin_limit.clamp(min=1.0e-6)
        ang_ratio = ang / ang_limit.clamp(min=1.0e-6)
        joint_ratio = joint / joint_limit.clamp(min=1.0e-6)
        # Give dense credit after the pose-lock delay. A reciprocal form avoids
        # exponential underflow while the robot is still rotating quickly,
        # preserving a useful learning signal. ``settled`` remains an
        # evaluation-only diagnostic, not a third training phase.
        contact_after_lock = self._pose_locked.float()
        residual_motion = lin_ratio.square() + ang_ratio.square() + joint_ratio.square()
        self._score = contact_after_lock / (1.0 + residual_motion)
        # Do not suppress the deliberately active impact-absorption phase.  The
        # residual-motion penalty starts when the natural pose is locked.
        self._motion_cost = contact_after_lock * (
            residual_motion
        )
        return self._settled.float()

    @property
    def settled(self) -> torch.Tensor:
        return self._settled

    @property
    def score(self) -> torch.Tensor:
        return self._score

    @property
    def motion_cost(self) -> torch.Tensor:
        return self._motion_cost

    @property
    def pose_drift(self) -> torch.Tensor:
        """Mean squared joint-position drift from the learned natural pose."""
        return self._pose_drift

    @property
    def action_drift(self) -> torch.Tensor:
        """Mean squared policy-target drift after the natural pose is locked."""
        return self._action_drift

    @property
    def phase(self) -> torch.Tensor:
        """Training-only phase label: 0 for impact, 1 for post-impact hold."""
        return self._pose_locked.long()

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._peak_lin[env_ids] = 0.0
        self._peak_ang[env_ids] = 0.0
        self._peak_joint[env_ids] = 0.0
        self._contact_seen[env_ids] = False
        self._contact_steps[env_ids] = 0
        self._stable_steps[env_ids] = 0
        self._settled[env_ids] = False
        self._pose_locked[env_ids] = False
        self._joint_pose_ref[env_ids] = 0.0
        self._action_ref[env_ids] = 0.0
        self._pose_drift[env_ids] = 0.0
        self._action_drift[env_ids] = 0.0
        self._score[env_ids] = 0.0
        self._motion_cost[env_ids] = 0.0


def post_impact_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the critic-only SafeFall phase label with shape ``[B, 1]``.

    This term is deliberately excluded from actor observations.  The shared
    recurrent actor infers the phase from IMU/joint-history dynamics, whereas
    the training critic uses the label to select its value head.
    """
    # ObservationManager probes term shapes before MetricsManager constructs
    # the accumulator.  Return the reset/impact phase during that probe.
    if not hasattr(env, "_safefall_settling_motion"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    phase = env._safefall_settling_motion.phase
    return phase.unsqueeze(-1).float()
