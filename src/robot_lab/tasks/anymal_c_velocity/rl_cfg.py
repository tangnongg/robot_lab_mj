"""RL configuration for ANYmal C velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def anymal_c_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for ANYmal C velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      obs_normalization=False,
      activation="elu",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      obs_normalization=False,
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name="anymal_c_velocity",
    max_iterations=10_000,
  )
