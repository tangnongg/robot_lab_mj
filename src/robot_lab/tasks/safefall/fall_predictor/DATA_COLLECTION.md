# Data Collection for Fall Predictor Training

> 论文 §III-B: *Data Collection* + Table I: *Technical Causes of Humanoid Falls*

---

## 1. 核心流程

```
  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  Nominal     │     │  扰动注入         │     │  摔倒检测         │
  │  Policy      │────▶│  (1-3 个因子)     │────▶│  base_h < 0.15m  │
  │  (速度跟踪)   │     │  每步记录 63-D    │     │  → 保存轨迹       │
  └──────────────┘     └──────────────────┘     └──────────────────┘
```

每步从 MuJoCo 仿真状态**直接读取传感器原始数据**（不依赖 observation tensor 格式），逐帧累积。支持多 env 并行采集以提升吞吐。当 base 高度低于 0.15 m 时判定为落地，episode 结束。只有真正发生了摔跤的 episode 才保存。

---

## 2. 用法

**前提**：先有训练好的 G1 velocity 策略 checkpoint。

```bash
conda activate env_mjlab_e
cd /home/tangl/myProj/robot_lab_mj

# 单 env 小规模测试
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 500 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/2026-06-11_01-03-02/model_1999.pt \
    --policy-task Mjlab-Velocity-Flat-Unitree-G1 \
    --device cuda:0

# 64 env 并行采集（高吞吐）
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 5000 \
    --num-envs 64 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --device cuda:0

# 粗粝地形（绊倒因子）
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs_rough \
    --num-trajs 5000 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --rough \
    --device cuda:0

# 带可视化窗口
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 500 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --viewer \
    --num-envs 64 \
    --device cuda:0

# 论文规模（约 23 GPU 小时单 env，多 env 大幅缩短）
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 81920 \
    --num-envs 4096 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/2026-06-11_01-03-02/model_1999.pt \
    --device cuda:0
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | (必需) | 轨迹 `.pt` 输出目录 |
| `--num-trajs` | 1000 | 目标轨迹数量 |
| `--num-envs` | 1 | 并行 env 数 |
| `--max-steps` | 500 | 单条 episode 最大步数 |
| `--device` | cpu | Torch 设备 (`cpu` / `cuda:0`) |
| `--seed` | 42 | 随机种子 |
| `--policy-checkpoint` | (必需) | Nominal policy 的 `.pt` 文件路径 |
| `--policy-task` | `Mjlab-Velocity-Flat-Unitree-G1` | Nominal policy 的 task ID |
| `--rough` | off | 使用粗粝地形（绊倒因子） |
| `--checkpoint-interval` | 50 | 每 N 条打印详细快照 |
| `--viewer` | off | 打开 MuJoCo 渲染窗口 |

---

## 3. 每步记录的数据

直接从 MuJoCo 实体读取（`extract_predictor_input`），**不依赖任何 env 的 observation 格式**：

```python
# 来自 model.py — _extract_from_entity()

# 1. 骨盆滚转/俯仰角 — 从根部四元数 (wxyz) 解算
quat = asset.data.root_link_quat_w      # [B, 4]
roll  = atan2(2(wx+yz), 1-2(x²+y²))    # rad
pitch = asin(clamp(2(wy-zx), -1, 1))   # rad
→ pelvis: [B, 2]

# 2. 根部角速度（body frame）
ang_vel = asset.data.root_link_ang_vel_b  # [B, 3]  rad/s

# 3. 关节位置（相对默认站立姿态）
jpos = asset.data.joint_pos - asset.data.default_joint_pos  # [B, 29]  rad

# 4. 关节速度
jvel = asset.data.joint_vel  # [B, 29]  rad/s

# 拼接 → [B, 63]
torch.cat([pelvis, ang_vel, jpos, jvel], dim=-1)
```

| 维度 | 字段 | 符号 | 来源 |
|------|------|------|------|
| 0-1 | pelvis roll, pitch | r_t | IMU (root quat) |
| 2-4 | base angular velocity | ω_t | IMU (gyro) |
| 5-33 | joint positions (relative) | q_t | 关节编码器 |
| 34-62 | joint velocities | q̇_t | 关节编码器 |

**与论文的差异**：mjlab G1 有 29 个关节（vs 论文 23 DoF URDF），输入为 63 维而非 51 维。

---

## 4. 扰动协议

每个 episode 随机选取 **1-3 个**扰动因子，完整实现了论文 Table I 的 6 种失效模式：

| # | 因子 | 实现方式 | 状态 |
|---|------|---------|------|
| 1 | 传感器噪声 | Observation 噪声 2-10× | ✅ |
| 2 | 外力推搡 | torso velocity perturbation: x∈[-2,2], y∈[-1,1] m/s | ✅ |
| 3 | 脚底打滑 | 随机脚踝速度扰动 =1.5 m/s | ✅ |
| 4 | 绊倒 | 粗粝地形 (`--rough`, 纯 heightfield) | ✅ |
| 5 | 系统延迟 | FIFO pipeline delay [20, 200] ms | ✅ |
| 6 | 动力学失配 | PD gain logU + CoM offset | ✅ |

### 4.1 传感器噪声

```python
sensor_noise_scale = uniform(2.0, 10.0)  # 2-10× 训练配置的量级
actor_obs += randn_like(actor_obs) * 0.01 * sensor_noise_scale
```

对 policy 输入层的 observation 加噪，模拟真实传感器漂移/故障。

### 4.2 外力推搡

```python
vx = uniform(-2.0, 2.0)   # m/s 前进/后退方向   ← 论文 Table I 原文
vy = uniform(-1.0, 1.0)   # m/s 侧向
apply_at_step = uniform(20, 80)
```

论文的 velocity perturbation to torso——直接覆写根部线速度，在随机时刻（第 20-80 步）施加。

### 4.3 脚底打滑

```python
slip_vel = uniform(-1.5, 1.5)   # m/s 水平速度
direction = uniform(0, 2π)       # 随机方向
apply_at_step = uniform(20, 80)
```

在随机时刻对随机脚踝 body 施加瞬时水平速度扰动。

### 4.4 系统延迟

```python
delay_steps = uniform(1, 11)  # 1-10 步 = 20-200 ms @ 50Hz
# FIFO 管线延迟：policy 每帧都在跑（看到最新 observation），
# 但 action 压入队列，delay_steps 帧后才弹出执行。
```

模拟真实系统中的 sensing→control→actuation 延迟链。与"action hold"（僵住不动）不同，policy 始终在工作，只是执行的是 N 步前的指令。

### 4.5 动力学失配

论文参数：
- Joint stiffness: `logU(0.7, 1.5) × nominal`
- Joint damping: `logU(0.5, 3.0) × nominal`
- CoM offset: `x,y ~ U(-0.05, 0.05)`, `z ~ U(-0.01, 0.01)`

实现方式（per-env，startup 时应用）：

```python
cfg.pd_gain_scale = (
    exp(uniform(log(0.7), log(1.5))),   # log-uniform → skews toward 1.0
    exp(uniform(log(0.5), log(3.0))),
)
cfg.com_offset = (
    uniform(-0.05, 0.05),  # x
    uniform(-0.05, 0.05),  # y
    uniform(-0.01, 0.01),  # z
)
```

PD 增益和 CoM 偏移在整个 episode 期间保持不变，下个 episode 重新随机化。

### 4.6 绊倒（需 `--rough`）

论文参数：unseen terrain heightfields，障碍物高度 [0, 15] cm。

实现方式：使用 `--rough` 标志，保留 flat policy 和 flat observation 空间，仅将地形替换为**论文参数规格**的自定义 terrain generator。三种地形类型：

| 地形类型 | 占比 | 说明 |
|---------|------|------|
| `random_rough` | 40% | 随机噪声高度场，noise [2, 15] cm |
| `discrete_obstacles` | 30% | 离散障碍块，高度 [3, 15] cm |
| `waves` | 30% | 正弦波地形，振幅 [3, 15] cm |

5×5 格 grid，每格 8×8 m，10m 扁平边界。**不需要 rough policy**——policy 始终是 flat 的（无 `height_scan`），在看不见的障碍物上更容易摔倒。

---

## 5. 并行采集架构

当 `--num-envs > 1` 时，采集循环自动切换为批式并行模式：

```
while saved < target:
    actions = policy_fn(obs)            # (N, 29) 一次前向
    for each env: per‑env FIFO delay    # 各 env 独立延迟
    for each env: per‑env perturbation  # 各 env 独立扰动（外力/脚滑）
    env.step(actions)                   # (N, 29) 一次物理步
    for each env: fall / max‑steps check
    for finished envs: save + reset     # 自动 recycle
```

Per-env 独立状态（`_EnvState`）：每个 env 有自己的 `TrajectoryWriter`、`PerturbConfig`、延迟队列。单 env（`--num-envs 1`）和之前的串行行为完全一致。

### Viewer 兼容

`--viewer` 在多 env 模式下也正常工作，但仅渲染 **env 0**（其余 env 在后台静默运行）。渲染所有 env 会受限于 MuJoCo 的 `mjv_addGeoms` 无法复制 heightfield 的顶点颜色——多 env 合成时地形会全白。只渲染 env 0 保证颜色正确、高度场细节可见。

---

## 6. 摔倒检测与保存条件

```python
# 每步检测
base_h = asset.data.root_link_pos_w[i, 2].item()
if base_h < 0.15 and step >= 3 * _T2_OFFSET_STEPS + 2:
    fell = True    # 落地完成

# 最低长度要求（满足时序分割的最小 T）
_MIN_TRAJ_LEN = 3 * _T2_OFFSET_STEPS + 2  # = 17 步
```

**不保存的情况**：
- Robot 始终没有摔倒（policy 太强，扰动太弱）
- 轨迹太短 < 17 步（无法做时序分割）
- episode 被 `time_out` / `joint_vel_exceeded` 终止但没摔倒

这意味着实际运行的 episode 数通常远大于 `--num-trajs`。

---

## 7. 轨迹文件格式

每条轨迹一个独立 `.pt` 文件：

```python
d = torch.load("traj_000001.pt", map_location="cpu", weights_only=False)

{
    "observations": torch.Tensor,  # [T, 63] float32
    "labels":       torch.Tensor,  # [T]    int64
    "T":            int,           # 轨迹总步数
}

# 标签含义
#   0  = safe      — t ≤ 2T/3
#  -1  = ambiguous — 2T/3 < t ≤ T-5步（训练时 mask）
#   1  = falling   — t > T-5步（落地前 100ms）
```

---

## 8. 数据规模参考

| 场景 | --num-trajs | --num-envs | 大约耗时 | 数据量 | 用途 |
|------|------------|-----------|---------|--------|------|
| 小规模测试 | 500 | 1 | ~30 min (cuda) | ~7 MB | 验证流水线 |
| 中等规模 | 5,000 | 1 | ~5 h (cuda) | ~75 MB | 测试训练收敛 |
| 高吞吐 | 5,000 | 64 | ~5 min (cuda) | ~75 MB | 快速迭代 |
| 论文规模 | 81,920 | 1 | ~23 h (cuda) | ~1.2 GB | 正式训练 |
| 论文规模+并行 | 81,920 | 4096 | ~20 min (cuda) | ~1.2 GB | 正式训练 |

---

## 9. 典型问题

### 编译慢

首次 `--device cuda:0` 运行会触发 MuJoCo-Warp JIT 编译（~30s-1min），输出类似：

```
Module _efc_contact_jac_sparse ... took 142.58 ms  (compiled)
Module mujoco_warp._src.ray ... took 774.93 ms    (compiled)
```

编译结果缓存在 `~/.cache/warp/`，之后再运行只需微秒级加载 `(cached)`。

### 采集速率

影响采集速率的因素：
- **摔倒率**：policy 越强（训练步数多），机器人越不容易摔
- **并行度**：`--num-envs` 越大吞吐越高（受 GPU 显存限制）
- **Warp kernel 编译**：仅首次运行

### 轨迹全部被拒绝

可能原因：
1. `max_steps` 太小，episode 结束时还没摔
2. Policy 太强，摔倒率极低 → 增大扰动强度
3. base_height 阈值设置不当 → 当前为 0.15 m，适合 G1

---

## 10. 代码参考

| 文件 | 内容 |
|------|------|
| `collect_data.py` | CLI + 采集循环（单/多 env） |
| `model.py` | `extract_predictor_input()` — 原始传感器数据提取 |
| `dataset.py` | `TrajectoryWriter`, `compute_labels()`, `load_trajectory()` |
| `train_predictor.py` | 使用采集数据训练 GRU predictor |
| `deploy.py` | `FallPredictorWrapper` — 部署时使用训练好的 predictor |
