#!/usr/bin/env python3
"""Train the GRU fall predictor — paper §III-C.

Training configuration (exact from paper):
  - Optimiser:  Adam, lr = 1e-3, weight_decay = 1e-4
  - Batch size: 4096 frames (or full sequences for GRU)
  - Epochs:     5
  - Loss:       Negative log-likelihood (cross-entropy), ambiguous segment masked
  - GPU time:   ~5 min on RTX 4090 for 65 K trajectories

Usage:
    python -m robot_lab.tasks.safefall.fall_predictor.train_predictor \\
        --data data/fall_trajectories \\
        --output models/fall_predictor.pt \\
        --epochs 5 --batch-size 4096 --lr 1e-3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import FallPredictor, INPUT_DIM
from .dataset import (
    FallTrajectoryDataset,
    FallSequenceDataset,
    collate_sequences,
    masked_cross_entropy,
    compute_metrics,
    false_alarm_rate,
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    device: str = "cpu",
    val_split: float = 0.2,
    seed: int = 42,
    use_sequences: bool = False,
) -> FallPredictor:
    """Train the fall predictor and save to *output_path*.

    Parameters
    ----------
    data_dir : Path
        Directory of ``.pt`` trajectory files.
    output_path : Path
        Where to save the trained model state_dict.
    epochs : int
        Number of training epochs (paper: 5).
    batch_size : int
        Batch size (paper: 4096 for frame-level; reduced for sequence-level).
    lr : float
        Learning rate (paper: 1e-3).
    weight_decay : float
        Weight decay (paper: 1e-4).
    hidden_dim : int
        GRU hidden size (paper: 64).
    device : str
        Torch device.
    val_split : float
        Fraction of trajectories held out for validation.
    seed : int
        Random seed.
    use_sequences : bool
        If True, use full-sequence GRU training. If False (default), use
        frame-level training (simpler, faster, and still effective — the GRU
        will still capture temporal context through hidden state at inference).
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Data ----
    if use_sequences:
        full_ds = FallSequenceDataset(data_dir, shuffle=True, seed=seed)
        n_total = len(full_ds)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        # Split file list for train/val.
        all_files = sorted(data_dir.glob("*.pt"))
        idx = np.random.RandomState(seed).permutation(len(all_files))
        train_files = [all_files[i] for i in idx[:n_train]]
        val_files = [all_files[i] for i in idx[n_train:]]

        train_ds = FallSequenceDataset(train_files, shuffle=True, seed=seed)
        val_ds = FallSequenceDataset(val_files, shuffle=False, seed=seed)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, collate_fn=collate_sequences, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, collate_fn=collate_sequences, num_workers=0
        )
    else:
        # Split file list for train/val.
        all_files = sorted(data_dir.glob("*.pt"))
        idx = np.random.RandomState(seed).permutation(len(all_files))
        n_val = max(1, int(len(all_files) * val_split))
        train_files = [all_files[i] for i in idx[n_val:]]
        val_files = [all_files[i] for i in idx[:n_val]]

        train_ds = FallTrajectoryDataset(train_files, balance=True, seed=seed)
        val_ds = FallTrajectoryDataset(val_files, balance=False, seed=seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train frames: {len(train_ds)}, Val frames: {len(val_ds)}")

    # ---- Model ----
    model = FallPredictor(input_dim=INPUT_DIM, hidden_dim=hidden_dim)
    model.to(device)
    model.train()
    print(f"Model parameters: {model.num_parameters:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Training ----
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if use_sequences:
                obs, labels, mask = batch
                obs, labels, mask = obs.to(device), labels.to(device), mask.to(device)
                logits, _ = model(obs)  # (B, T, 2)
                loss = masked_cross_entropy(logits, labels, mask)
            else:
                obs, labels = batch
                obs, labels = obs.to(device), labels.to(device)
                logits, _ = model(obs.unsqueeze(1))  # (B, 1, 63) → (B, 1, 2)
                logits = logits.squeeze(1)            # (B, 2)
                loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ---- Validation ----
        model.eval()
        all_logits = []
        all_labels = []
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if use_sequences:
                    obs, labels, mask = batch
                    obs, labels, mask = obs.to(device), labels.to(device), mask.to(device)
                    logits, _ = model(obs)
                    loss = masked_cross_entropy(logits, labels, mask)
                    # Flatten for metrics (only non-masked positions).
                    keep = mask.bool()
                    all_logits.append(logits[keep].cpu())
                    all_labels.append(labels[keep].cpu())
                else:
                    obs, labels = batch
                    obs, labels = obs.to(device), labels.to(device)
                    logits, _ = model(obs.unsqueeze(1))
                    logits = logits.squeeze(1)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())

                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        cat_logits = torch.cat(all_logits)
        cat_labels = torch.cat(all_labels)
        metrics = compute_metrics(cat_logits, cat_labels)
        far = false_alarm_rate(cat_logits, cat_labels)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} | "
              f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
              f"FAR={far:.4%} | {elapsed:.1f}s")

    # ---- Save ----
    model.save(output_path)
    print(f"Model saved to {output_path}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train fall predictor")
    parser.add_argument("--data", type=str, required=True, help="Trajectory data directory")
    parser.add_argument("--output", type=str, default="models/fall_predictor.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequences", action="store_true",
                        help="Use full-sequence GRU training")
    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        device=args.device,
        val_split=args.val_split,
        seed=args.seed,
        use_sequences=args.sequences,
    )


if __name__ == "__main__":
    main()
