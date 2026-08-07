"""RL configuration for SafeFall G1 task.

PPO hyperparameters matching the SafeFall paper (Meng et al., 2025).
Migrated from RslRlPpoActorCriticCfg + RslRlPpoAlgorithmCfg.

Paper Section III-D / III-E:
- Asymmetric actor-critic: actor sees deployable observations; critic
  additionally accesses privileged state (root pos/vel, CoM).
- 3-layer MLP for both actor and critic.
- Adam optimizer, lr=1e-3, adaptive schedule.
- Fixed episode length of 40 steps at 50 Hz.
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def unitree_g1_safefall_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree G1 SafeFall task.

    Matches the IsaacLab SafeFallPPORunnerCfg hyperparameters:
    - Actor: [256, 256, 128] with ELU (deployable observations)
    - Critic: [512, 256, 128] with ELU (privileged observations)
    - PPO with adaptive schedule, 40 steps per env, 5000 max iterations
    - Asymmetric: actor uses "actor" obs group, critic uses "critic" obs group
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            # Phase is inferred from IMU/joint dynamics by this recurrent actor;
            # no contact or phase label is passed at inference time.
            class_name="RNNModel",
            rnn_type="gru",
            rnn_hidden_dim=128,
            rnn_num_layers=1,
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                # The previous run drove this to the 2.0 ceiling, producing
                # aggressive impact actions and severe leg folding.
                "init_std": 0.5,
                "std_range": (5.0e-2, 0.75),
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            # Two independent value heads selected from the critic-only phase
            # group.  The shared actor remains phase-agnostic.
            class_name="robot_lab.tasks.safefall.two_phase_rl:TwoPhaseCritic",
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="safefall_g1",
        # Stage I runs can be interrupted or inspected frequently.  Preserve
        # each 100-update milestone for regression/video evaluation.
        save_interval=100,
        num_steps_per_env=40,                   # Paper: 40 steps per episode
        max_iterations=5000,
        clip_actions=10.0,  # Absolute targets are clipped per joint by the env.
        # Asymmetric actor-critic (paper Section III-D):
        #   actor  → deployable sensor measurements (noisy)
        #   critic → actor terms (clean) + privileged root pos/vel/CoM
        obs_groups={
            "actor": ("actor",),
            "critic": ("critic", "critic_phase"),
        },
    )
