# Temporal Segmentation & Masked Loss — Design Rationale

Paper: SafeFall §III-C (Meng et al., 2025)

---

## 1. Problem

A fall trajectory has three phases.  The transition from "safe" to "inevitable falling" is gradual — there is no single timestep where the switch happens.  Labelling the transition region as either safe or falling teaches the model the wrong mapping.

```
t:   0 ────────────────── t1=2T/3 ──────── t2=T-100ms ──── T (ground impact)
      │                      │                 │              │
      SAFE                   │              FALLING           │
      robot walking          │         impact inevitable      │
      or recovering          │                                │
                             │
                    AMBIGUOUS ZONE
                    could go either way
```

Paper: "This transition is gradual, not instantaneous, and depends on the robot's complex dynamics and initial conditions."

---

## 2. Why not drop ambiguous frames

Two dataset classes handle this differently, and the choice matters.

### Frame-level dataset (`FallTrajectoryDataset`)

Drops ambiguous frames at load time:

```python
keep = labels >= 0          # keep only safe(0) and falling(1)
all_obs.append(obs[keep])
```

This is safe here because each frame is an independent training sample — no temporal dependency between consecutive frames.  Dropping two frames in the middle of a trajectory has no side effect; DataLoader shuffles everything anyway.

### Sequence dataset (`FallSequenceDataset`)

**Keeps** ambiguous frames (label = −1) and masks their loss instead:

```python
yield d["observations"], d["labels"]   # labels contain -1
```

Why it must keep them: the GRU hidden state propagates forward through every timestep.  If frames t=16,17 are removed:

```
Drop t=16,17:
  x_15 → h_15 → ??? → h_18 → h_19 → ...
          ↑               ↑
     safe context    suddenly falling

  GRU loses two hidden-state updates.
  The transition from safe to falling happens in a single discontinuous jump.
```

Keep t=16,17 with masked loss:

```
Keep t=16,17 (masked):
  x_15 → h_15 → x_16 → h_16 → x_17 → h_17 → x_18 → h_18 → x_19 → ...
  │              │              │              │              │
  ∇loss_15      ∇=0           ∇=0           ∇loss_18      ∇loss_19
  │              │              │              │              │
  backprop      masked         masked        backprop      backprop
                from loss      from loss
```

The gradient of `loss_18` backpropagates **through** `h_17` and `h_16` — the chain `∂loss_18/∂h_18 × ∂h_18/∂h_17 × ∂h_17/∂h_16 × ∂h_16/∂h_15` is unbroken.  GRU weights are updated across the transition zone even though those positions contribute no direct loss.  The model learns what the hidden state *should look like* across the transition without being forced to output a specific label there.

---

## 3. Why not use `ignore_index=-1` in `cross_entropy`

`F.cross_entropy(..., ignore_index=-100)` handles one ignored value.  But collated sequences have **two** distinct types of excluded positions:

| Position type | Label value | Origin |
|--------------|-------------|--------|
| Ambiguous frames | −1 | Paper temporal segmentation |
| Padding | −100 | `collate_sequences` (tail beyond trajectory length) |

`ignore_index` can only be set to one value.  Explicit masking decouples the two:

```python
# collate_sequences builds the mask:
valid = labels >= 0              # 1 for safe/falling, 0 for ambiguous
mask[i, :T] = valid.float()      # 0 for tail positions (padding)

# masked_cross_entropy ignores both:
loss = F.cross_entropy(logits, labels, reduction="none")
return (loss * mask).sum() / mask.sum()   # mask=0 → no gradient
```

The mask is the single source of truth for which positions participate in training, independent of label values.

---

## 4. Forward vs backward in the GRU

| Direction | Ambiguous frames | GRU weights |
|----------|-----------------|-------------|
| Forward pass | Full sequence — h_t updates normally | Same as any other timestep |
| Backward pass | Loss contribution ×0 | Updated via gradients flowing **through** these positions from later falling frames |
| What the model learns | Hidden state that bridges safe→falling | Parameters that process the transition correctly |

The model doesn't learn "what label should be output here" — it learns "how the hidden state should evolve to make subsequent predictions correct."

---

## 5. Summary

| Aspect | Frame-level (`FallTrajectoryDataset`) | Sequence-level (`FallSequenceDataset`) |
|--------|--------------------------------------|----------------------------------------|
| Returns | Single frame `(63,)` | Full trajectory `(T, 63)` |
| Ambiguous frames | Dropped at load | Kept, loss masked to 0 |
| Temporal context | None (shuffled) | GRU hidden state propagates through all frames |
| Gradient flow across transition | N/A | Continuous — transition zone weights updated |
| Use | Fast validation, smoke tests | Proper GRU training |
