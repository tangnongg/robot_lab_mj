"""Deployment wrapper for real-time fall prediction — paper §III-C + §III-D.

The wrapper runs the GRU predictor at each timestep alongside either:
  1. A nominal locomotion policy (normal operation), or
  2. The SafeFall mitigation policy (after a fall is detected).

Paper deployment flow (Fig. 2d):
  - Predictor continuously monitors proprioceptive observations.
  - When falling probability > threshold → switch to SafeFall policy.
  - Predictor hidden state is maintained across timesteps.

Key paper metrics:
  - Inference time: < 0.5 ms (onboard CPU)
  - False alarm rate: < 0.1% (paper Table III)
  - Lead time: 410 ms (paper Table III, t2 = T - 100 ms config)

Usage::

    from robot_lab.tasks.safefall.fall_predictor import (
        FallPredictor, FallPredictorWrapper, extract_predictor_input,
    )

    model = FallPredictor()
    model.load("models/fall_predictor.pt")
    wrapper = FallPredictorWrapper(model, threshold=0.5, device="cpu")

    for step in range(max_steps):
        obs_dict, _, _, _, _ = env.step(action)

        is_falling, prob = wrapper.update(obs_dict)
        if is_falling:
            # Switch to SafeFall policy.
            action = safefall_policy(obs_dict)
        else:
            action = nominal_policy(obs_dict)
"""

from __future__ import annotations

import torch

from .model import FallPredictor, extract_predictor_input


class FallPredictorWrapper:
    """Lightweight wrapper that maintains GRU hidden state across timesteps.

    Parameters
    ----------
    model : FallPredictor
        Trained GRU predictor.
    threshold : float
        Probability above which a fall is declared (default 0.5).
    device : str
        Torch device for inference.
    warmup_steps : int
        Number of initial steps to suppress predictions (allows GRU state
        to stabilise from zero-init).
    confirmation_steps : int
        Number of consecutive "falling" predictions required before
        triggering.  Hysteresis to reduce false positives.
    """

    def __init__(
        self,
        model: FallPredictor,
        threshold: float = 0.5,
        device: str = "cpu",
        warmup_steps: int = 10,
        confirmation_steps: int = 3,
        env: object | None = None,
    ):
        self.model = model.to(device)
        self.threshold = threshold
        self.device = device
        self.warmup_steps = warmup_steps
        self.confirmation_steps = confirmation_steps
        self._env = env  # optional, for entity-based extraction

        self._hidden: torch.Tensor | None = None
        self._step_count: int = 0
        self._consecutive_falling: int = 0
        self._is_falling: bool = False
        self._last_prob: float = 0.0
        self._t_detect: int = 0

    def reset(self) -> None:
        """Reset hidden state and counters (call at episode start)."""
        self._hidden = None
        self._step_count = 0
        self._consecutive_falling = 0
        self._is_falling = False
        self._last_prob = 0.0
        self._t_detect = 0

    def update(self, obs_dict: dict[str, torch.Tensor]) -> tuple[bool, float]:
        """Process one observation frame and return (is_falling, probability).

        Call this once per environment step.  Maintains internal GRU state
        and implements hysteresis to reduce false triggers.

        Parameters
        ----------
        obs_dict : dict
            The observation dict returned by ``env.step()``.  Must contain
            the ``"actor"`` key.

        Returns
        -------
        is_falling : bool
            True if a fall has been detected (latched until :meth:`reset`).
        probability : float
            Current falling probability (0-1).
        """
        self._step_count += 1

        # Extract predictor input from entity data (paper: IMU + joint encoders).
        x = extract_predictor_input(self._env).to(self.device)  # (B, 63)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Forward pass.
        with torch.no_grad():
            prob, self._hidden = self.model.predict_proba(x, self._hidden)

        self._last_prob = float(prob.item() if prob.numel() == 1 else prob[0].item())

        # Warmup: suppress predictions.
        if self._step_count <= self.warmup_steps:
            return False, self._last_prob

        # Strict consecutive‑frame hysteresis.
        # A single frame below threshold resets the counter to zero.
        if self._last_prob > self.threshold:
            self._consecutive_falling += 1
        else:
            self._consecutive_falling = 0

        if self._consecutive_falling >= self.confirmation_steps:
            if not self._is_falling:
                # The alarm is raised NOW.  Record this moment so the
                # caller can compute Lead Time = t_impact - _t_detect.
                self._t_detect = self._step_count
            self._is_falling = True  # latch — stays raised until reset()

        return self._is_falling, self._last_prob

    @property
    def is_falling(self) -> bool:
        """Whether a fall has been detected (latched)."""
        return self._is_falling

    @property
    def probability(self) -> float:
        """Most recent falling probability."""
        return self._last_prob

    @property
    def lead_time_steps(self) -> int:
        """Steps elapsed since the alarm was raised.

        Lead Time (paper §III-C / Table III): the interval between
        the fall predictor raising the alarm (``_is_falling`` first
        becoming ``True``) and ground impact.  Read this property at
        impact time to get the full lead time in steps (* 20 ms).

        A longer lead time gives the SafeFall mitigation policy more
        margin to execute protective maneuvers (paper reports 410 ms
        average for the t₂ = T-100 ms config).
        """
        if not self._is_falling:
            return 0
        return self._step_count - self._t_detect

    @property
    def detection_step(self) -> int:
        """Global step at which the alarm was first raised (latch point).

        Callers can compute: ``t_impact - detection_step`` to obtain
        the paper's Lead Time metric.
        """
        return self._t_detect


def load_predictor(
    checkpoint_path: str,
    threshold: float = 0.5,
    device: str = "cpu",
    env: object | None = None,
    **kwargs,
) -> FallPredictorWrapper:
    """Load a trained predictor from checkpoint and return a wrapped instance.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``.pt`` checkpoint file saved by :meth:`FallPredictor.save`.
    threshold : float
        Detection threshold.
    device : str
        Torch device.
    env : ManagerBasedRlEnv or None
        Optional environment instance for entity-based input extraction.
    kwargs :
        Passed to :class:`FallPredictorWrapper`.

    Returns
    -------
    FallPredictorWrapper
    """
    model = FallPredictor()
    model.load(checkpoint_path)
    model.eval()
    return FallPredictorWrapper(model, threshold=threshold, device=device, env=env, **kwargs)
