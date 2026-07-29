"""RL configuration for the G1 HoST stand-up task.

Five PPO runner variants — one per starting pose (ground, platform, wall,
slope, prone). All share the same network architecture and algorithm
hyperparameters; only the experiment name and iteration count differ.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_g1_host_ppo_runner_cfg(
    experiment_name: str = "g1_host_ground",
    max_iterations: int = 12000,
) -> RslRlOnPolicyRunnerCfg:
    """Build a PPO runner config for the HoST stand-up task.

    Args:
        experiment_name: Name used for logging / checkpoint directories.
        max_iterations: Maximum number of PPO iterations.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256),
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
        experiment_name=experiment_name,
        save_interval=100,
        num_steps_per_env=50,
        max_iterations=max_iterations,
        clip_actions=1.0,
        logger="tensorboard",
        upload_model=False,
    )


# ---------------------------------------------------------------------------
# Per-variant convenience constructors
# ---------------------------------------------------------------------------


def unitree_g1_host_ground_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return unitree_g1_host_ppo_runner_cfg("g1_host_ground", 12000)


def unitree_g1_host_platform_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return unitree_g1_host_ppo_runner_cfg("g1_host_platform", 12000)


def unitree_g1_host_wall_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return unitree_g1_host_ppo_runner_cfg("g1_host_wall", 7500)


def unitree_g1_host_slope_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return unitree_g1_host_ppo_runner_cfg("g1_host_slope", 10000)


def unitree_g1_host_prone_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return unitree_g1_host_ppo_runner_cfg("g1_host_prone", 12000)
