"""Phase-conditioned double critic for the HoST stand-up task.

The policy is shared across the whole episode, while the value function has
two independent heads:

* ``standup_critic`` is used before the robot reaches the standing gate.
* ``post_task_critic`` is used after the gate and learns the standing/holding
  objective.

The phase bit is present only in the critic observation group.  It is therefore
available during training, but is not part of the deployable actor input.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

class HoSTTwoPhaseCritic(MLPModel):
    """Two independent value heads selected by the critic-only phase bit."""

    _phase_obs_group = "critic_phase"
    num_heads = 2

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims=(512, 256),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    ):
        if output_dim != 1:
            raise ValueError(f"HoSTTwoPhaseCritic requires scalar output, got {output_dim}")
        if distribution_cfg is not None:
            raise ValueError("HoSTTwoPhaseCritic is deterministic and cannot use a distribution")
        if self._phase_obs_group not in obs_groups[obs_set]:
            raise ValueError(
                f"{self._phase_obs_group!r} must be included in the {obs_set!r} observation set"
            )
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg=None,
        )
        del self.mlp
        self.standup_critic = MLP(self._get_latent_dim(), 1, hidden_dims, activation)
        self.post_task_critic = MLP(self._get_latent_dim(), 1, hidden_dims, activation)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        latent = self.get_latent(obs)
        standup_value = self.standup_critic(latent)
        post_task_value = self.post_task_critic(latent)
        phase = obs[self._phase_obs_group][..., 0].to(dtype=torch.bool)
        return torch.where(phase.unsqueeze(-1), post_task_value, standup_value)


# Backward-compatible name for callers that imported the old HoST critic.
HoSTMultiCritic = HoSTTwoPhaseCritic


# Historical algorithm import paths are aliases only.  The canonical task uses
# the stock PPO class directly, so there is no stage-specific warm-start logic.
HoSTPhasePPO = PPO
HoSTMultiCriticPPO = PPO
