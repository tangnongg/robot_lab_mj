"""Four-value-head PPO for the HoST stand-up task.

The actor remains a single policy.  The critic predicts four independent value
functions, one for task completion, regularization, style, and post-stand hold
rewards.  Each head gets its own GAE target; normalized advantages are mixed for
the actor update so the hold objective cannot be washed out by the dense task
reward.
"""

from __future__ import annotations

from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict


HOST_REWARD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("task_orientation", "task_head_height"),
    (
        "regu_dof_acc", "regu_action_rate", "regu_smoothness", "regu_torques",
        "regu_joint_power", "regu_dof_vel", "regu_joint_tracking_error",
        "regu_dof_pos_limits", "regu_dof_vel_limits",
    ),
    (
        "style_waist_deviation", "style_hip_yaw_deviation", "style_hip_roll_deviation",
        "style_shoulder_roll_deviation", "style_left_foot_displacement",
        "style_right_foot_displacement", "style_knee_deviation", "style_shank_orientation",
        "style_ground_parallel", "style_feet_distance", "style_ang_vel_xy",
    ),
    (
        "target_ang_vel_xy", "target_lin_vel_xy", "target_feet_height_var",
        "target_upper_dof_pos", "target_orientation", "target_base_height",
    ),
)


class HoSTMultiCritic(MLPModel):
    """Four independent deterministic value networks over the critic input."""

    num_heads = 4

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256),
                 activation="elu", obs_normalization=False, distribution_cfg=None):
        if output_dim != 1 or distribution_cfg is not None:
            raise ValueError("HoSTMultiCritic expects a scalar deterministic critic config")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg=None)
        del self.mlp
        self.critics = torch.nn.ModuleList(
            [MLP(self._get_latent_dim(), 1, hidden_dims, activation) for _ in range(self.num_heads)]
        )

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output=False):
        del stochastic_output
        latent = self.get_latent(obs, masks, hidden_state)
        return torch.cat([head(latent) for head in self.critics], dim=-1)


class HoSTMultiCriticStorage(RolloutStorage):
    """Rollout storage with four-dimensional critic values and group rewards."""

    def __init__(self, *args, num_heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = torch.zeros(
            self.num_transitions_per_env, self.num_envs, num_heads, device=self.device
        )
        self.returns = torch.zeros_like(self.values)
        self.group_rewards = torch.zeros_like(self.values)


class HoSTMultiCriticPPO(PPO):
    """PPO with independent GAE per HoST reward group."""

    group_weights = (0.25, 0.35, 0.65, 1.50)

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        storage = HoSTMultiCriticStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device,
            num_heads=HoSTMultiCritic.num_heads,
        )
        algorithm = alg_class(
            actor, critic, storage, device=device, **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        algorithm.env = env
        return algorithm

    def __init__(self, actor, critic, storage, *args, **kwargs):
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.group_weights_tensor = torch.tensor(self.group_weights, device=self.device).view(1, 1, -1)
        self._group_indices: list[list[int]] | None = None

    def _get_group_rewards(self) -> torch.Tensor:
        manager = self.env.unwrapped.reward_manager
        if self._group_indices is None:
            names = manager.active_terms
            self._group_indices = [[names.index(name) for name in group if name in names] for group in HOST_REWARD_GROUPS]
        term_reward = manager._step_reward * self.env.unwrapped.step_dt
        return torch.stack(
            [term_reward[:, indices].sum(dim=1) if indices else torch.zeros(term_reward.shape[0], device=term_reward.device)
             for indices in self._group_indices], dim=-1
        )

    def process_env_step(self, obs, rewards, dones, extras):
        group_rewards = self._get_group_rewards().detach()
        time_outs = extras.get("time_outs")
        if time_outs is not None:
            group_rewards = group_rewards + self.gamma * self.transition.values * time_outs.unsqueeze(-1).to(self.device)
        self.storage.group_rewards[self.storage.step].copy_(group_rewards)

        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if time_outs is not None:
            self.transition.rewards += self.gamma * self.transition.values.mean(dim=-1) * time_outs.to(self.device)
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs):
        st = self.storage
        last_values = self.critic(obs).detach()
        advantages = torch.zeros_like(st.values)
        gae = torch.zeros(st.num_envs, HoSTMultiCritic.num_heads, device=self.device)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            not_done = 1.0 - st.dones[step].float()
            delta = st.group_rewards[step] + not_done * self.gamma * next_values - st.values[step]
            gae = delta + not_done * self.gamma * self.lam * gae
            advantages[step] = gae
            st.returns[step] = gae + st.values[step]
        normalized = (advantages - advantages.mean(dim=(0, 1), keepdim=True)) / (advantages.std(dim=(0, 1), keepdim=True) + 1e-8)
        mixed = (normalized * self.group_weights_tensor).sum(dim=-1, keepdim=True)
        st.advantages.copy_((mixed - mixed.mean()) / (mixed.std() + 1e-8))
