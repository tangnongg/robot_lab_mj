#!/usr/bin/env python3
"""Train the GRU fall predictor — paper §III-C.

Training configuration (exact from paper):
  - Data:      full trajectories, ambiguous segment masked
  - Optimiser: Adam, lr = 1e-3, weight_decay = 1e-4
  - Epochs:    5
  - Loss:      masked cross-entropy
  - GPU time:  ~5 min on RTX 4090 for 65 K trajectories

Usage:
    python -m robot_lab.tasks.safefall.fall_predictor.train_predictor \\
        --data data/fall_trajectories \\
        --output models/fall_predictor.pt \\
        --epochs 5 --batch-size 32 --lr 1e-3
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
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    device: str = "cpu",
    val_split: float = 0.2,
    seed: int = 42,
) -> FallPredictor:
    """Train the fall predictor and save to *output_path*."""
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Data: full-sequence GRU training ----
    all_files = sorted(data_dir.glob("*.pt"))
    if not all_files:
        raise FileNotFoundError(f"No .pt trajectory files found in {data_dir}")

    idx = np.random.RandomState(seed).permutation(len(all_files))
    n_val = max(1, int(len(all_files) * val_split))
    n_train = len(all_files) - n_val
    train_files = [all_files[i] for i in idx[:n_train]]
    val_files = [all_files[i] for i in idx[n_train:]]

    train_ds = FallSequenceDataset(train_files, shuffle=True, seed=seed)
    val_ds = FallSequenceDataset(val_files, shuffle=False, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        collate_fn=collate_sequences, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        collate_fn=collate_sequences, num_workers=0,
    )

    print(f"Train trajectories: {len(train_ds)}, Val: {len(val_ds)}")

    # ---- Model ----
    model = FallPredictor(input_dim=INPUT_DIM, hidden_dim=hidden_dim)
    model.to(device)
    model.train()
    print(f"Model parameters: {model.num_parameters:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Training ----
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            obs, labels, mask = batch
            obs = obs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            logits, _ = model(obs)  # (B, T, 2)
            loss = masked_cross_entropy(logits, labels, mask)

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
                obs, labels, mask = batch
                obs = obs.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                logits, _ = model(obs)
                loss = masked_cross_entropy(logits, labels, mask)

                # Flatten for metrics (only non-masked positions).
                keep = mask.bool()
                all_logits.append(logits[keep].cpu())
                all_labels.append(labels[keep].cpu())

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
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/fall_predictor.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of trajectories per batch (sequences vary in length)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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
    )


if __name__ == "__main__":
    main()
