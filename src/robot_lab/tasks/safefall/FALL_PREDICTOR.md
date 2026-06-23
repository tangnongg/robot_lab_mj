# Fall Predictor — Implementation & Usage Guide

> Paper: **SafeFall: Learning Protective Control for Humanoid Robots** (Meng et al., 2025)
> Section III-C: *Fall Predictor*

---

## 1. Architecture

```
Input (63-D)             GRU (64 hidden)           Output (2-class)
─────────────────       ───────────────        ──────────────────
r_t   (2)  pelvis roll,pitch     ┌─┐
ω_t   (3)  base ang vel          │G│ 64 units       Linear(64→2)
q_t  (29)  joint pos (rel)       │R│ ───────►      ─────────────►  [safe, falling]
q̇_t (29)  joint vel              │U│  single layer
                   └─┘
```

- **Parameters**: ~25 K (larger than paper's ~13 K due to 29-DOF MJCF vs 23-DOF URDF)
- **Inference**: < 1 ms on CPU
- **Training**: Full-sequence GRU with masked cross-entropy (paper §III-C)
- **Deployment**: Runs continuously alongside nominal policy; only the predictor executes during normal operation

---

## 2. Temporal Segmentation

Paper defines three phases in each falling trajectory of length T:

```
timestep:  0 ─────────────── t1=2T/3 ─────── t2=T-100ms ──── T (ground impact)
           │                   │                  │              │
label:     SAFE (0)          AMBIGUOUS (-1)     FALLING (1)
trained:   ✓                  ✗ (masked)         ✓
```

- **Safe** (t ≤ t1): Includes recoverable instabilities — prevents false alarms during aggressive maneuvers
- **Ambiguous** (t1 < t ≤ t2): Transition region — masked from loss, preventing the model from learning uncertain labels
- **Falling** (t > t2): 100 ms window before impact — minimum reliable detection horizon

Detailed rationale in [TEMPORAL_SEGMENTATION.md](TEMPORAL_SEGMENTATION.md).

---

## 3. Files

```
safefall/fall_predictor/
├── __init__.py              # Package exports
├── model.py                 # FallPredictor GRU model + extract_predictor_input
├── dataset.py               # FallSequenceDataset, collate_sequences, masked_cross_entropy
├── collect_data.py          # Multi-env data collection (nominal policy + 6 perturbations)
├── train_predictor.py       # GRU sequence training (masked cross-entropy, Adam)
├── deploy.py                # FallPredictorWrapper for real-time deployment
├── DATA_COLLECTION.md       # Detailed data collection guide
├── TEMPORAL_SEGMENTATION.md # Why ambiguous frames are masked, not dropped
├── TENSOR_BASICS.md         # Shape conventions reference
└── ABLATION_PLAN.md         # Warmup/hysteresis ablation experiment plan
```

---

## 4. Quickstart

### Step 1: Train a Nominal Policy (or use an existing checkpoint)

```bash
python -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 4096 --agent.max-iterations 2000
```

### Step 2: Collect Training Data

```bash
conda activate env_mjlab_e
cd /home/tangl/myProj/robot_lab_mj

# Small-scale test
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 500 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --device cuda:0

# Multi-env parallel (high throughput)
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 5000 --num-envs 64 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --device cuda:0

# With rough terrain (foot trip factor)
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs_rough --num-trajs 5000 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --rough --device cuda:0

# With viewer window
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs --num-trajs 500 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --viewer
```

See [DATA_COLLECTION.md](DATA_COLLECTION.md) for full details.

### Step 3: Train the Predictor

```bash
python -m robot_lab.tasks.safefall.fall_predictor.train_predictor \
    --data data/fall_trajs \
    --output models/fall_predictor.pt \
    --epochs 5 --batch-size 32 --lr 1e-3 \
    --device cuda:0
```

Training uses full-sequence GRU with masked cross-entropy (paper §III-C). `--batch-size` is number of trajectories per batch (sequences vary in length, padded by `collate_sequences`).

Expected output:
```
Train trajectories: 4000, Val: 1000
Model parameters: 24,898
Epoch 1/5 | train_loss=0.4231 val_loss=0.3102 | acc=0.8723 prec=0.8912 rec=0.8501 f1=0.8701 FAR=0.12%
...
Epoch 5/5 | train_loss=0.1821 val_loss=0.1853 | acc=0.9423 prec=0.9501 rec=0.9321 f1=0.9410 FAR=0.04%
Model saved to models/fall_predictor.pt
```

**Paper baseline** (Table III, t1=2T/3, t2=T-100ms):
- False Alarm Rate: 0.06%
- Lead Time: 410 ms

### Step 4: Deploy

```python
from robot_lab.tasks.safefall.fall_predictor import load_predictor

predictor = load_predictor(
    "models/fall_predictor.pt",
    threshold=0.5,
    device="cpu",
    warmup_steps=10,         # suppress predictions during initial steps
    confirmation_steps=3,    # require 3 consecutive positives (hysteresis)
)

for step in range(max_steps):
    obs_dict, _, _, _, _ = env.step(action)
    is_falling, prob = predictor.update(obs_dict)

    if is_falling:
        action = safefall_policy(obs_dict)
    else:
        action = nominal_policy(obs_dict)

    if terminated or truncated:
        predictor.reset()
        obs_dict, _ = env.reset()
```

---

## 5. API Reference

### `FallPredictor` (model.py)

```python
model = FallPredictor(input_dim=63, hidden_dim=64, num_layers=1)

# Single frame
logits, h_next = model(x)              # x: (B, 63) → logits: (B, 2)

# Sequence (training)
logits, h_next = model(x_seq)          # x_seq: (B, T, 63) → (B, T, 2)

# Persistence
model.save("path.pt")
model.load("path.pt")
```

### `FallPredictorWrapper` (deploy.py)

```python
wrapper = FallPredictorWrapper(
    model, threshold=0.5, device="cpu",
    warmup_steps=10, confirmation_steps=3,
)

is_falling, prob = wrapper.update(obs_dict)  # per-step call
wrapper.is_falling     # bool, latched once triggered
wrapper.probability    # float, latest falling probability
wrapper.reset()        # call at episode start
```

### `load_predictor` (convenience)

```python
predictor = load_predictor("models/fall_predictor.pt", threshold=0.5, device="cpu")
```

### Data collection

```python
from robot_lab.tasks.safefall.fall_predictor import collect_trajectories

collect_trajectories(
    env_cfg=env_cfg,
    output_dir="data/fall_trajs",
    num_trajs=5000,
    policy_fn=policy_fn,
    num_envs=64,         # parallel envs
    max_steps=500,
    device="cuda:0",
)
```

### Training

```python
from robot_lab.tasks.safefall.fall_predictor import train

model = train(
    data_dir="data/fall_trajs",
    output_path="models/predictor.pt",
    epochs=5, batch_size=32, lr=1e-3,
    device="cuda:0",
)
```

---

## 6. Perturbation Protocol (Paper Table I)

Each trajectory randomly activates 1–3 of the following factors:

| # | Factor | Implementation | Status |
|---|--------|---------------|--------|
| 1 | Sensor Noise | Observation noise 2–10× | ✅ |
| 2 | External Force | Velocity perturbation to torso: x∈[−2,2], y∈[−1,1] m/s | ✅ |
| 3 | Foot Slip | Horizontal velocity kick to random foot, ±1.5 m/s | ✅ |
| 4 | Foot Trip | Paper-spec custom heightfield terrain (`--rough` flag) | ✅ |
| 5 | System Delay | FIFO pipeline delay [20, 200] ms | ✅ |
| 6 | Dynamic Mismatch | PD gain logU scale + CoM offset | ✅ |

See [DATA_COLLECTION.md §4](DATA_COLLECTION.md) for implementation details of each factor.

---

## 7. Deployment Integration with SafeFall

The complete SafeFall deployment (paper Fig. 2d):

```
                    ┌────────────────────┐
                    │   Fall Predictor    │  runs every step
                    │   (GRU, 64 hidden)  │  < 1 ms inference
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  is_falling?        │
                    └─────────┬──────────┘
                              │
              ┌───────────────┴───────────────┐
              │ NO                            │ YES
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │  Nominal Policy   │           │  SafeFall Policy  │
    │  (locomotion)     │           │  (damage mitigation)│
    └──────────────────┘           └──────────────────┘
              │                               │
              └───────────┬───────────────────┘
                          ▼
                  ┌──────────────┐
                  │  PD Controller│  200 Hz
                  │  (G1 joints)  │
                  └──────────────┘
```

The predictor is the **only** component running during normal operation. The SafeFall policy stays dormant until triggered.

---

## 8. Known Limitations

1. **Input dimension**: mjlab G1 has 29 DOF (vs paper 23), giving 63-D input instead of 51-D.
2. **No Stage II curriculum**: The predictor is trained standalone; full Stage II requires feeding predictor output back into the SafeFall policy training loop.
3. **PD gain / CoM DR may cause VRAM growth** with large `--num-envs` due to Warp graph re-capture. Currently applied per-episode; this is a known issue being investigated.
4. **Warmup and hysteresis** are pragmatic additions not described in the paper. An ablation experiment is planned ([ABLATION_PLAN.md](ABLATION_PLAN.md)) to quantify their necessity.
5. **CoM offset** values are sampled but may not apply correctly in all multi-env configurations.

---

## 9. Shape Conventions

See [TENSOR_BASICS.md](TENSOR_BASICS.md) for a full reference on tensor shapes used throughout this module.

| Tensor | Shape |
|--------|-------|
| Single-frame observation | `(B, 63)` |
| Full trajectory | `(T, 63)` |
| Padded batch | `(B, T_max, 63)` |
| GRU hidden state | `(1, B, 64)` |
| Classification logits | `(B, T, 2)` |

---

## 10. References

- Paper: https://arxiv.org/abs/2511.18509
- Project page: https://safefall.github.io
- GRU: Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", arXiv:1412.3555, 2014.
