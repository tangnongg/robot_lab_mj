"""Lightweight GRU-based fall predictor — paper §III-C.

Architecture (paper exact):
  - Input:  s_t = {r_t, ω_t, q_t, q̇_t}  (63 dims)
    - r_t   pelvis roll, pitch (world frame) ................. 2
    - ω_t   base angular velocity ............................. 3
    - q_t   joint positions (relative to default stance) ...... 29
    - q̇_t   joint velocities .................................. 29
  - GRU:   single layer, 64 hidden units
  - Head:  Linear(64 → 2)  → binary logits (safe / falling)
  - Inference: < 0.5 ms (paper claim)
  - Params: ~13 K

The predictor maintains its own hidden state, updated at each
timestep.  This is a *stateless from the outside* wrapper that
hides the state-management details.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


# ---------------------------------------------------------------------------
# Input dimension breakdown (used for sanity checks and subset extraction).
# ---------------------------------------------------------------------------
INPUT_DIMS = {
    "pelvis_orientation": 2,   # roll, pitch
    "base_ang_vel": 3,
    "joint_pos": 29,           # rel to default
    "joint_vel": 29,
}
INPUT_DIM = sum(INPUT_DIMS.values())  # 63


# ---------------------------------------------------------------------------
# GRU model
# ---------------------------------------------------------------------------

class FallPredictor(nn.Module):
    """Single-layer GRU fall predictor matching paper §III-C.

    Parameters
    ----------
    input_dim : int
        Observation dimension (default 63).
    hidden_dim : int
        GRU hidden size (paper: 64).
    num_layers : int
        GRU layers (paper: 1).
    dropout : float
        Dropout after GRU (paper doesn't mention; 0 by default).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, 2)  # 2-class logits

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, input_dim) or (B, input_dim)
            Observation sequence.  If 2-D a time dimension of 1 is added.
        h : (num_layers, B, hidden_dim) or None
            Initial hidden state.  If None, zero-initialised.

        Returns
        -------
        logits : (B, T, 2) or (B, 2)
            Raw class scores.  ``logits[..., 1]`` corresponds to **falling**.
        h_next : (num_layers, B, hidden_dim)
            Final hidden state (can be fed into the next call).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, input_dim) → (B, 1, input_dim)
            squeeze_out = True
        else:
            squeeze_out = False

        if h is None:
            h = torch.zeros(
                self.num_layers, x.size(0), self.hidden_dim,
                device=x.device, dtype=x.dtype,
            )

        out, h_next = self.gru(x, h)          # out: (B, T, hidden)
        logits = self.classifier(out)          # (B, T, 2)

        if squeeze_out:
            logits = logits.squeeze(1)         # (B, 2)

        return logits, h_next

    def predict_proba(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return falling probability and updated hidden state.

        Convenience wrapper around :meth:`forward`.
        """
        logits, h_next = self.forward(x, h)
        probs = torch.softmax(logits, dim=-1)
        return probs[..., 1], h_next   # falling probability, next hidden

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Observation extraction helpers
# ---------------------------------------------------------------------------

def extract_predictor_input(env: object) -> torch.Tensor:
    """Extract 63-D predictor input directly from MuJoCo entity data.

    Reads raw sensor values — works with ANY env (velocity, safefall, etc.)
    regardless of observation format.  Matches the paper's description that
    the predictor reads "onboard IMU and joint encoders."

    Parameters
    ----------
    env : ManagerBasedRlEnv
        Environment with ``scene`` access to entity data.

    Returns
    -------
    x : (B, 63)
        [pelvis_rp(2), ang_vel(3), jpos(29), jvel(29)]
    """
    return _extract_from_entity(env)


def _extract_from_entity(env: object) -> torch.Tensor:
    """Extract predictor input directly from the MuJoCo simulation state.

    Works with ANY env (velocity, safefall, etc.) — no dependency on
    observation term layout.  Matches paper §III-C exactly.
    """
    from mjlab.entity import Entity
    asset: Entity = env.scene["robot"]

    # Pelvis roll, pitch from root quaternion (wxyz).
    quat = asset.data.root_link_quat_w  # (B, 4)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    pelvis = torch.stack([roll, pitch], dim=-1)  # (B, 2)

    # Base angular velocity (body frame).
    ang_vel = asset.data.root_link_ang_vel_b  # (B, 3)

    # Joint positions relative to default stance.
    default_pos = asset.data.default_joint_pos
    jpos = asset.data.joint_pos - default_pos  # (B, 29)

    # Joint velocities.
    jvel = asset.data.joint_vel  # (B, 29)

    return torch.cat([pelvis, ang_vel, jpos, jvel], dim=-1)  # (B, 63)
