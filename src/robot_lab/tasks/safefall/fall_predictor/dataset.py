"""Trajectory dataset with paper temporal segmentation for fall predictor training.

Paper §III-C segmentation:
  For each trajectory of length T (where T is ground-impact time):
    t1 = 2T/3                          — end of safe segment
    t2 = T - 100 ms (= T - 5 steps,
         因为Policy运行在50Hz)    — start of falling segment
    ─────────────────────────────────────────────────────
    t ≤ t1          → safe     (label 0, trained)
    t1 < t ≤ t2     → ambiguous (label -1, masked from loss)
    t > t2          → falling  (label 1, trained)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .model import INPUT_DIM, extract_predictor_input

# Steps at 50 Hz corresponding to 100 ms.
_T2_OFFSET_STEPS = 5  # 100 ms ÷ 0.02 s


# ---------------------------------------------------------------------------
# Temporal segmentation
# ---------------------------------------------------------------------------

def compute_labels(
    traj_len: int,
    t2_offset: int = _T2_OFFSET_STEPS,
) -> torch.Tensor:
    """Return per-timestep labels for a trajectory of length *traj_len*.

    Returns
    -------
    labels : (T,) int64
        0 = safe, 1 = falling, −1 = ambiguous (masked).
    """
    T = traj_len
    if T <= 3 * t2_offset + 2:
        raise ValueError(
            f"Trajectory too short (length {T}); should not happen in "
            "practice."
        )

    t1 = int(np.ceil(2 * T / 3))              
    t2 = T - t2_offset           

    # Sanity check: ensure t1 < t2 < T to avoid label conflicts.
    if not (0 < t1 < t2 < T):
        raise ValueError(
            f"Invalid segmentation for trajectory length {T}: "
            f"t1={t1}, t2={t2}"
        )

    labels = torch.full((T,), -1, dtype=torch.long)  # ambiguous by default
    labels[:t1] = 0    # safe
    labels[t2:] = 1    # falling
    return labels


# ---------------------------------------------------------------------------
# Trajectory on-disk format
# ---------------------------------------------------------------------------

class TrajectoryWriter:
    """Accumulate per-step observations and flush to disk as a single trajectory.

    Usage::

        writer = TrajectoryWriter()
        for obs_dict, ... in rollout:
            writer.add(obs_dict)
        writer.save(output_dir / "traj_0001.pt")
    """

    def __init__(self, env: object | None = None, env_idx: int = 0) -> None:
        self._frames: list[torch.Tensor] = []
        self._root_states: list[torch.Tensor] = []
        self._joint_positions: list[torch.Tensor] = []
        self._joint_velocities: list[torch.Tensor] = []
        self._env = env
        self._env_idx = env_idx

    def add(self) -> None:
        """Add one frame by extracting 63-D predictor input from entity data.

        For multi‑env collection prefer ``add_from_batch`` which extracts
        once for all envs instead of N separate GPU allocations.
        """
        x = extract_predictor_input(self._env).detach().cpu()  # (B, 63)
        if x.dim() == 2:
            x = x[self._env_idx].clone()
        self._frames.append(x)

        asset = self._env.scene["robot"]
        root_state = torch.cat(
            [
                asset.data.root_link_pos_w,
                asset.data.root_link_quat_w,
                asset.data.root_link_vel_w,
            ],
            dim=-1,
        ).detach().cpu()
        root_state[:, 0:3] -= self._env.scene.env_origins.detach().cpu()
        self._root_states.append(root_state[self._env_idx].clone())
        self._joint_positions.append(
            asset.data.joint_pos[self._env_idx].detach().cpu().clone()
        )
        self._joint_velocities.append(
            asset.data.joint_vel[self._env_idx].detach().cpu().clone()
        )

    def add_from_batch(
        self,
        batch: torch.Tensor,
        root_state_batch: torch.Tensor | None = None,
        joint_pos_batch: torch.Tensor | None = None,
        joint_vel_batch: torch.Tensor | None = None,
    ) -> None:
        """Add a frame from a pre-extracted (B, 63) CPU tensor.

        *batch* must already be on CPU and detached.  This avoids N
        redundant GPU→CPU transfers when collecting with many envs.
        """
        self._frames.append(batch[self._env_idx].clone())
        state_batches = (root_state_batch, joint_pos_batch, joint_vel_batch)
        if all(value is None for value in state_batches):
            return
        if any(value is None for value in state_batches):
            raise ValueError(
                "Full trajectory state requires root, joint-pos, and joint-vel batches"
            )
        assert root_state_batch is not None
        assert joint_pos_batch is not None
        assert joint_vel_batch is not None
        self._root_states.append(root_state_batch[self._env_idx].clone())
        self._joint_positions.append(joint_pos_batch[self._env_idx].clone())
        self._joint_velocities.append(joint_vel_batch[self._env_idx].clone())

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def tensor(self) -> torch.Tensor:
        """Full observation sequence, shape (T, 63)."""
        return (
            torch.stack(self._frames)
            if self._frames
            else torch.empty(0, INPUT_DIM)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "observations": self.tensor,          # tensor (T, 63)
            "labels": compute_labels(len(self)),  # tensor (T,)
            "T": len(self),  # int （后续的使用不涉及矩阵运算；int 占用少量内存）
        }
        if len(self._root_states) == len(self._frames):
            data.update(
                {
                    "root_state": torch.stack(self._root_states),
                    "joint_pos": torch.stack(self._joint_positions),
                    "joint_vel": torch.stack(self._joint_velocities),
                    "state_schema_version": 2,
                }
            )
        torch.save(data, path)


def load_trajectory(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a single trajectory ``.pt`` file."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Iterable dataset for training on full sequences (GRU-compatible)
# ---------------------------------------------------------------------------

class FallSequenceDataset(IterableDataset):
    """Yields full trajectory sequences for GRU training.

    Each item is ``(observations, labels)`` where
      - observations: (T, 63)  — full proprioceptive sequence
      - labels:       (T,)     — per-timestep labels (-1 = masked)

    The ambiguous frames are kept (label = -1) so that the training loop
    can mask them from the loss computation.
    """

    def __init__(
        self,
        data_dir: str | Path | list[str] | list[Path],
        shuffle: bool = True,
        seed: int = 42,
    ):
        if isinstance(data_dir, list):
            self.files = [Path(f) for f in data_dir]
        else:
            data_dir = Path(data_dir)
            self.files = sorted(data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No .pt trajectory files found in {data_dir}"
            )
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        files = list(self.files)
        if self.shuffle:
            self.rng.shuffle(files)
        for f in files:
            d = load_trajectory(f)
            yield d["observations"], d["labels"].long()

    def __len__(self) -> int:
        return len(self.files)


# ---------------------------------------------------------------------------
# Collation and masking helpers
# ---------------------------------------------------------------------------

def collate_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to create a batch.

    Returns
    -------
    obs_padded : (B, T_max, 63)
    labels_padded : (B, T_max)
    mask : (B, T_max)  — 1.0 for timesteps to include in loss, 0.0 for pad + ambiguous.
    """
    B = len(batch)
    T_max = max(obs.shape[0] for obs, _ in batch)

    obs_padded = torch.zeros(B, T_max, INPUT_DIM)
    labels_padded = torch.full((B, T_max), -100, dtype=torch.long)  # ignore_index
    mask = torch.zeros(B, T_max)

    for i, (obs, labels) in enumerate(batch):
        T = obs.shape[0]
        obs_padded[i, :T] = obs
        labels_padded[i, :T] = labels
        # Map ambiguous (-1) to cross_entropy's ignore_index so
        # the loss function skips them natively.
        labels_padded[i, :T][labels_padded[i, :T] == -1] = -100
        # Mask: include only non-ambiguous (label >= 0) timesteps.
        valid = labels >= 0
        mask[i, :T] = valid.float()

    return obs_padded, labels_padded, mask


def masked_cross_entropy(
    logits: torch.Tensor,   # (B, T, 2)
    labels: torch.Tensor,   # (B, T)
    mask: torch.Tensor,     # (B, T)
) -> torch.Tensor:
    """Cross-entropy loss computed only on masked positions.

    *labels* should contain -100 at ignored positions (PyTorch convention),
    but *mask* provides the explicit weight.
    """
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 2),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Quick validation utilities
# ---------------------------------------------------------------------------

def compute_metrics(
    logits: torch.Tensor,   # (N, 2)  or (B, T, 2) flattened
    labels: torch.Tensor,   # (N,)
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 for binary classification."""
    preds = logits.argmax(dim=-1)
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
    }


def false_alarm_rate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Paper metric: fraction of safe states misclassified as falling."""
    preds = logits.argmax(dim=-1)
    safe_mask = labels == 0
    if safe_mask.sum() == 0:
        return 0.0
    fp = ((preds == 1) & safe_mask).sum().item()
    return fp / safe_mask.sum().item()
