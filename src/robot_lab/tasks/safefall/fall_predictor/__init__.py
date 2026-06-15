"""Fall predictor — lightweight GRU-based binary classifier for fall detection.

Paper: SafeFall §III-C (Meng et al., 2025).

Components
----------
- :class:`FallPredictor` — single-layer GRU, 64 hidden units, ~13 K params.
- :class:`FallPredictorWrapper` — deployment wrapper with hysteresis and
  hidden-state management.
- :func:`extract_predictor_input` — extract 63-D input from env observation dict.
- :func:`collect_trajectories` — data collection using nominal policy + perturbations.
- :func:`train` — training loop with masked cross-entropy.

Quickstart
----------
1. Collect data::

    python -m robot_lab.tasks.safefall.fall_predictor.collect_data \\
        --output data/fall_trajs --num-trajs 100 --random

2. Train::

    python -m robot_lab.tasks.safefall.fall_predictor.train_predictor \\
        --data data/fall_trajs --output models/predictor.pt

3. Deploy::

    from robot_lab.tasks.safefall.fall_predictor import load_predictor
    predictor = load_predictor("models/predictor.pt")
    is_falling, prob = predictor.update(obs_dict)
"""

from .model import FallPredictor, extract_predictor_input, INPUT_DIM
from .deploy import FallPredictorWrapper, load_predictor
from .dataset import (
    TrajectoryWriter,
    compute_labels,
    FallTrajectoryDataset,
    FallSequenceDataset,
    collate_sequences,
    masked_cross_entropy,
    compute_metrics,
    false_alarm_rate,
)
from .collect_data import collect_trajectories
from .train_predictor import train
