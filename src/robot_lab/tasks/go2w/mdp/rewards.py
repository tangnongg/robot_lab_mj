"""Useful methods for MDP rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)
from mjlab.sensor import RayCastSensor, ContactSensor


if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

from mjlab.utils.lab_api.math import (
  quat_apply, 
  quat_apply_inverse,
  quat_conjugate
)


def lin_vel_z_l2(  
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_link_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(  
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # "root_com_ang_vel_b" equals to "root_link_ang_vel_b"
    reward = torch.sum(torch.square(asset.data.root_com_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    if sensor_name is not None:
        sensor: RayCastSensor = env.scene[sensor_name]
        # Adjust the target height using the sensor data
        # Use hit_pos_w (hit positions in world frame) instead of ray_hits_w
        ray_hits = sensor.data.hit_pos_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# TODO:
def body_lin_acc_l2(
  env: ManagerBasedRlEnv, 
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    # # no attri body_link_acc_w
    # return torch.sum(torch.norm(asset.data.body_link_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)
    return torch.tensor([0.0])

# TODO: not used in robot_lab, weight is 0
def joint_vel_limits(
  env: ManagerBasedRlEnv, 
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits."""
    asset: Entity = env.scene[asset_cfg.name]
    # # no attri soft_joint_vel_limits
    # soft_joint_vel_limits = asset.data.soft_joint_vel_limits
    # assert soft_joint_vel_limits is not None
    # out_of_limits = -(
    #   asset.data.joint_vel[:, asset_cfg.joint_ids]
    #   - soft_joint_vel_limits[:, asset_cfg.joint_ids, 0]
    # ).clip(max=0.0)
    # out_of_limits += (
    #   asset.data.joint_vel[:, asset_cfg.joint_ids]
    #   - soft_joint_vel_limits[:, asset_cfg.joint_ids, 1]
    # ).clip(min=0.0)
    # return torch.sum(out_of_limits, dim=1)
    return torch.tensor([0.0])

# same with electric_power_cost in mjlab
def joint_power(
  env: ManagerBasedRlEnv, 
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        # note: maybe should be actuator_force instead of qfrc_actuator
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.qfrc_actuator[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward

def stand_still_without_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def wheel_vel_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, env.scene[sensor_name].body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    # standing_reward = torch.sum(
    #     torch.where(joint_vel > 0.1, joint_vel, 0.0),  # 仅惩罚>0.1 rad/s的转动
    #     dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def wheel_vel_stand_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
)-> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    standing_reward = torch.sum(joint_vel, dim=1)
    # standing_reward = torch.sum(
    #     torch.where(joint_vel > 0.1, joint_vel, 0.0),  # 仅惩罚>0.1 rad/s的转动
    #     dim=1)
    reward = torch.where(
        cmd > command_threshold,
        0.0,
        standing_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_mirror(
  env: ManagerBasedRlEnv, 
  asset_cfg: SceneEntityCfg, 
  mirror_joints: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(
  env: ManagerBasedRlEnv, 
  asset_cfg: SceneEntityCfg, 
  mirror_joints: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(
  env: ManagerBasedRlEnv, 
  asset_cfg: SceneEntityCfg, 
  joint_groups: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# TODO: not used, weight is 0; no applied_torque, computed_torque
def applied_torque_limits(
  env: ManagerBasedRlEnv, 
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # # extract the used quantities (to enable type-hinting)
    # asset: Entity = env.scene[asset_cfg.name]
    # # compute out of limits constraints
    # # TODO: We need to fix this to support implicit joints.
    # out_of_limits = torch.abs(
    #     asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    # )
    # return torch.sum(out_of_limits, dim=1)
    return torch.tensor([0.0])
def undesired_contacts(env: ManagerBasedRlEnv, threshold: float, sensor_name: str) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    data = contact_sensor.data
    assert data.force_history is not None
    # print(f"[undesired_contacts reward] data.force_history shape is {data.force_history.shape}")
    force_mag = torch.norm(data.force_history, dim=-1) # [B N H] [4096, 13, 4] vs [N T B][4096, 3, 13] in robot_lab
    is_contact = torch.max(force_mag, dim=2)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# same with robot_lab
def contact_forces(env: ManagerBasedRlEnv, threshold: float, sensor_name: str) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    data = contact_sensor.data
    # print(f"[contact_forces reward] sensor_cfg.ids issssssssssssssssss: {sensor_cfg.body_ids}")
    # print(f"[contact_forces reward] sensor_cfg.names issssssssssssssssss: {sensor_cfg.body_names}")
    # compute the violation
    # net_contact_forces shape is [B N H 3]
    force_mag = torch.norm(data.force_history, dim=-1) # [B N H][4096, 4, 4] vs [N T B][4096, 3, 4] in robot_lab
    violation = torch.max(force_mag, dim=2)[0] - threshold # [B N]
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1) # [B]

# # robot_lab
# def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize contact forces as the amount of violations of the net contact force."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     net_contact_forces = contact_sensor.data.net_forces_w_history
#     # compute the violation
#     # net_contact_forces shape is [N T B 3]
#     force_mag = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1) # [N T B]
#     violation = torch.max(force_mag, dim=1)[0] - threshold # [N B]
#     # compute the penalty
#     return torch.sum(violation.clip(min=0.0), dim=1) # [N]

def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv, 
  std: float, 
  command_name: str, 
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute the error
    command = env.command_manager.get_command(command_name)
    assert command is not None
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv, 
  std: float, 
  command_name: str, 
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute the error
    command = env.command_manager.get_command(command_name)
    assert command is not None
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# TODO: 
def feet_air_time(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_name: str,
    mode_time: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    asset: Entity = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    assert contact_sensor.data.current_air_time is not None
    assert contact_sensor.data.current_contact_time is not None
    current_air_time = contact_sensor.data.current_air_time[:, env.scene[sensor_name]]
    current_contact_time = contact_sensor.data.current_contact_time[:, env.scene[sensor_name]]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    reward = torch.sum(reward, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# TODO:  Use uniree rl lab's feet gait here so far
def feet_gait(
  env: ManagerBasedRlEnv, 
  period: float, 
  offset: float,
  threshold: float,
  command_threshold: float,
  command_name: str, 
  sensor_name: str
) -> torch.Tensor:
    """Reward feet gait"""
    contact_sensor: ContactSensor = env.scene[sensor_name] 
    current_contact_time = contact_sensor.data.current_contact_time
    assert current_contact_time is not None
    is_contact = current_contact_time > 0.0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).squeeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = torch.remainder(global_phase + offsets, 1.0)
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact.float().mean(dim=1))
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            agular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + agular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_contact(
  env: ManagerBasedRlEnv, 
  command_name: str, 
  expect_contact_num: int, 
  sensor_name: str
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_contact_without_cmd(
  env: ManagerBasedRlEnv, 
  command_name: str, 
  sensor_name: str
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_stumble(
  env: ManagerBasedRlEnv, 
  sensor_name: str
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    force = contact_sensor.data.force
    assert force is not None
    forces_z = torch.abs(force[:, env.scene[sensor_name].body_ids, 2])
    forces_xy = torch.linalg.norm(force[:, env.scene[sensor_name].body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

## simalar to feet_slip in mjlab 
def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  cost *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
  return cost

def feet_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: Entity = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_height_body(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: Entity = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_link_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_link_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_link_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_distance_y_exp(
    env: ManagerBasedRlEnv, 
    stance_width: float, 
    std: float, 
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = quat_apply(
            quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def stand_still_wheel(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle_vel), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def upward(
    env: ManagerBasedRlEnv, 
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward