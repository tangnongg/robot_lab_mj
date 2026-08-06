"""Training-only two-head critic for the implicit-phase SafeFall policy.

The actor never receives a phase label.  A separate ``critic_phase``
observation is visible only to this value model and selects one of two value
heads: impact protection (phase 0) or post-impact holding (phase 1).  Returning
the selected scalar keeps the standard RSL-RL PPO/GAE implementation valid:
the next state's phase automatically selects the correct bootstrap head at the
phase transition.
"""

from __future__ import annotations

import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict


class TwoPhaseCritic(MLPModel):
    """Two independent value heads selected by a critic-only phase signal."""

    _phase_obs_group = "critic_phase"

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        if output_dim != 1:
            raise ValueError(f"TwoPhaseCritic requires scalar value output, got {output_dim}")
        if distribution_cfg is not None:
            raise ValueError("TwoPhaseCritic is deterministic and cannot use a distribution")
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

        # Do not share the value heads: their targets have different reward
        # scales and horizons.  The common observation normalizer remains
        # shared, but the critics themselves are independent MLPs.
        del self.mlp
        self.impact_critic = MLP(self._get_latent_dim(), 1, hidden_dims, activation)
        self.hold_critic = MLP(self._get_latent_dim(), 1, hidden_dims, activation)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state, stochastic_output
        if masks is not None:
            obs = unpad_trajectories(obs, masks)

        latent = self.get_latent(obs)
        impact_value = self.impact_critic(latent)
        hold_value = self.hold_critic(latent)
        phase = obs[self._phase_obs_group][..., 0].to(dtype=torch.bool)
        return torch.where(phase.unsqueeze(-1), hold_value, impact_value)
