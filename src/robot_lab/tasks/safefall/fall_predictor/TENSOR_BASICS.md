# PyTorch Tensor Shape Conventions in This Project

## 1. `(B, T, D)` — the standard sequence format

Most tensors carrying temporal data use three dimensions:

| Dim | Symbol | Meaning | Example |
|-----|--------|---------|--------|
| 0 | B | Batch — number of independent trajectories in one GPU pass | 8 |
| 1 | T | Time — number of timesteps (varies per trajectory) | 23..328 |
| 2 | D | Feature dimension (fixed per feature type) | 63, 2, 3, 29 |

```python
# Observations: (B, T, 63)  — 63-D proprioceptive input per timestep
# GRU output:   (B, T, 64)  — 64-D hidden state per timestep
# Logits:       (B, T, 2)   — 2-class scores per timestep
# Mask:         (B, T)      — scalar weight per timestep
```

## 2. `(B, D)` — the single-frame format

When there is no temporal dimension:

```python
# Frame-level dataloader: each item is (63,) → batch (4096, 63)
# GRU input (single-frame inference): (1, 63)
# Hidden state: (num_layers, B, 64)
```

## 3. Common operations

### View / Reshape

`view` changes shape without copying data.  The total number of elements must be identical.

```python
x = torch.randn(50, 480)        # (B=50, 5*96=480) — flattened history from safefall env
x.view(50, 5, 96)               # (B, H=5, D=96) — one slice per history frame
x.view(50, 5, 96)[:, -1, :]     # (B, 96) — latest frame
```

### Unsqueeze / Squeeze

Add or remove a dimension of size 1:

```python
x = torch.randn(4096, 63)       # (B, D) — frame batch
x.unsqueeze(1)                  # (B, 1, D) — insert T=1 for single-step GRU forward

x = torch.randn(1, 2)           # (1, 2) — single frame logits
x.squeeze(0)                    # (2,) — remove batch dim
```

### Cat

Stack along an existing dimension:

```python
a = torch.randn(100, 63)   # 100 frames from traj_0
b = torch.randn(200, 63)   # 200 frames from traj_1
c = torch.randn(150, 63)   # 150 frames from traj_2

torch.cat([a, b, c], dim=0)  # (450, 63) — all frames concatenated
```

### Stack

Create a new dimension from identically-shaped tensors:

```python
a = torch.randn(23, 63)    # traj_0
b = torch.randn(23, 63)    # traj_1 — must be same shape as a

torch.stack([a, b], dim=0) # (2, 23, 63) — new batch dim
```

`stack` requires all tensors to have identical shape.  For variable-length sequences, use `cat` in a loop with padding instead.

## 4. Shape conventions by file

| File | Key tensors | Shapes |
|------|------------|--------|
| `model.py` | `extract_predictor_input` return | `(B, 63)` |
| `model.py` | `FallPredictor.forward` input | `(B, T, 63)` or `(B, 63)` |
| `model.py` | `FallPredictor.forward` output | `(B, T, 2)` or `(B, 2)` |
| `model.py` | GRU hidden state | `(num_layers, B, hidden_dim)` |
| `dataset.py` | `TrajectoryWriter.tensor` | `(T, 63)` — single trajectory |
| `dataset.py` | `FallTrajectoryDataset` item | `(63,)` — single frame |
| `dataset.py` | `FallSequenceDataset` item | `(T, 63)` — full trajectory |
| `dataset.py` | `collate_sequences` output | `(B, T_max, 63)`, `(B, T_max)`, `(B, T_max)` |
| `dataset.py` | `masked_cross_entropy` inputs | `(B, T, 2)`, `(B, T)`, `(B, T)` |
| `collect_data.py` | Saved `.pt` file `observations` | `(T, 63)` |
| `collect_data.py` | Saved `.pt` file `labels` | `(T,)` |
