"""Useful methods for MDP events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# def randomize_rigid_body_inertia(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | None,
#     asset_cfg: SceneEntityCfg,
#     inertia_distribution_params: tuple[float, float],
#     operation: Literal["add", "scale", "abs"],
#     distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
# ):
#     """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

#     This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
#     The function samples random values from the given distribution parameters and adds, scales, or sets the values
#     into the physics simulation based on the operation.

#     .. tip::
#         This function uses CPU tensors to assign the body inertias. It is recommended to use this function
#         only during the initialization of the environment.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]

#     # resolve environment ids
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device="cpu")
#     else:
#         env_ids = env_ids.cpu()

#     # resolve body indices
#     if asset_cfg.body_ids == slice(None):
#         body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
#     else:
#         body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

#     # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
#     inertias = asset.root_physx_view.get_inertias()

#     # apply randomization on default values
#     inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

#     # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
#     for idx in [0, 4, 8]:
#         # Extract and randomize the specific diagonal element
#         randomized_inertias = _randomize_prop_by_op(
#             inertias[:, :, idx],
#             inertia_distribution_params,
#             env_ids,
#             body_ids,
#             operation,
#             distribution,
#         )
#         # Assign the randomized values back to the inertia tensor
#         inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

#     # set the inertia tensors into the physics simulation
#     asset.root_physx_view.set_inertias(inertias, env_ids)