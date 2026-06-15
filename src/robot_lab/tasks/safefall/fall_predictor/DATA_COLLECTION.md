# Data Collection for Fall Predictor Training

> 论文 §III-B: *Data Collection* + Table I: *Technical Causes of Humanoid Falls*

---

## 1. 核心流程

```
  ┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  Nominal     │     │  扰动注入         │     │  摔倒检测         │
  │  Policy      │────▶│  (1-3 个因子)     │────▶│  base_h < 0.15m  │
  │  (速度跟踪)   │     │  每步记录 63-D    │     │  → 保存轨迹       │
  └─────────────┘     └──────────────────┘     └──────────────────┘
```

每步从 MuJoCo 仿真状态**直接读取传感器原始数据**（不依赖 observation tensor 格式），逐帧累积。当 base 高度低于 0.15 m 时判定为落地，episode 结束。只有真正发生了摔跤的 episode 才保存。

## 2. 用法

### Quick test（随机动作，无需 policy）

```bash
conda activate env_mjlab_e
cd /home/tangl/myProj/robot_lab_mj

python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 500 \
    --random \
    --device cpu
```

每一两步保存一条轨迹（falling init 状态本来就接近地面），500 条 2-3 分钟完成。

### 正式采集（velocity policy + 扰动）

**前提**：先有训练好的 G1 velocity 策略 checkpoint。

```bash
# 小规模测试
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 500 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/2026-06-11_01-03-02/model_1999.pt \
    --policy-task Mjlab-Velocity-Flat-Unitree-G1 \
    --device cuda:0

# 论文规模（约 23 GPU 小时）
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs \
    --num-trajs 81920 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --policy-task Mjlab-Velocity-Flat-Unitree-G1 \
    --device cuda:0
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | (必需) | 轨迹 `.pt` 输出目录 |
| `--num-trajs` | 1000 | 目标轨迹数量 |
| `--max-steps` | 500 | 单条 episode 最大步数 |
| `--device` | cpu | Torch 设备 (`cpu` / `cuda:0`) |
| `--seed` | 42 | 随机种子 |
| `--random` | off | 使用随机动作（不加载 policy） |
| `--policy-checkpoint` | None | Nominal policy 的 `.pt` 文件路径 |
| `--policy-task` | `Mjlab-Velocity-Flat-Unitree-G1` | Nominal policy 的 task ID |

## 3. 每步记录的数据

直接从 MuJoCo 实体读取（`_extract_from_entity`），**不依赖任何 env 的 observation 格式**：

```python
# 来自 model.py

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

# 拼接
input = cat([pelvis, ang_vel, jpos, jvel], dim=-1)  # [B, 63]
```

| 维度 | 字段 | 符号 | 来源 |
|------|------|------|------|
| 0-1 | pelvis roll, pitch | r_t | IMU (root quat) |
| 2-4 | base angular velocity | ω_t | IMU (gyro) |
| 5-33 | joint positions (relative) | q_t | 关节编码器 |
| 34-62 | joint velocities | q̇_t | 关节编码器 |

**与论文的差异**：mjlab G1 有 29 个关节（vs 论文 23 DoF URDF），输入为 63 维而非 51 维。

## 4. 扰动协议

每个 episode 随机选取 **1-3 个**扰动因子，完整实现了论文 Table I 的 6 种失效模式：

| # | 因子 | 实现方式 | 状态 |
|---|------|---------|------|
| 1 | 传感器噪声 | Observation 噪声 2-10× | ✅ |
| 2 | 外力推搡 | torso velocity perturbation: x∈[-2,2], y∈[-1,1] m/s | ✅ |
| 3 | 脚底打滑 | 随机脚踝速度扰动 1.5 m/s | ✅ |
| 4 | 绊倒 | 粗粝地形 (`--rough`, terrain generator) | ✅ |
| 5 | 系统延迟 | obs→action delay [20, 200] ms | ✅ |
| 6 | 动力学失配 | PD gain scale [0.2, 3.0]× + CoM offset | ✅ |

### 4.1 传感器噪声

```python
sensor_noise_scale = uniform(2.0, 10.0)  # 2-10× 训练配置的量级
actor_obs += randn_like(actor_obs) * 0.01 * sensor_noise_scale
```

对 policy 输入层的 observation 加噪，模拟真实传感器漂移/故障。

### 4.2 外力推搡

```python
vx = uniform(-2.0, 2.0)   # m/s 前进/后退方向
vy = uniform(-1.0, 1.0)   # m/s 侧向
# 转换为力脉冲施加到 torso (body 0)
force = (vx * 30, vy * 15, 0)  # N
apply_at_step = uniform(20, 80)
```

论文的 velocity perturbation to torso，在随机时刻施加。

### 4.3 脚底打滑

```python
slip_vel = uniform(-1.5, 1.5)   # m/s 水平速度
direction = uniform(0, 2π)       # 随机方向
apply_at_step = uniform(20, 80)
```

对随机选中的脚踝 body（left_ankle_roll_link / right_ankle_roll_link）施加瞬时水平速度。

### 4.4 系统延迟

```python
delay_steps = uniform(1, 11)  # 1-10 步 = 20-200 ms @ 50Hz
# 用环形 buffer 缓存最近 N 个 action，延迟输出
```

模拟真实系统中的 sensing→control→actuation 延迟链。

### 4.5 动力学失配（✅ 已实现）

论文参数：Joint stiffness `logU(0.7, 1.5) × nominal`，Joint damping `logU(0.5, 3.0) × nominal`，CoM offset `x,y ~ U(-0.05, 0.05)`, `z ~ U(-0.01, 0.01)`。

实现方式：

```python
# 每 episode 开始时调用 mjlab DR 函数
dr_actuator.pd_gains(env, env_ids=None,
    kp_range=(kp_scale, kp_scale),
    kd_range=(kd_scale, kd_scale),
    asset_cfg=SceneEntityCfg("robot", actuator_names=(".*",)),
    distribution="uniform", operation="scale",
)
dr_body.body_com_offset(env, env_ids=None,
    ranges={"x": (dx, dx), "y": (dy, dy), "z": (dz, dz)},
    asset_cfg=SceneEntityCfg("robot", body_names=("torso_link",)),
)
```

PD 增益缩放和 CoM 偏移在整个 episode 期间保持不变，下个 episode 重新随机化。

### 4.6 绊倒（✅ 已实现，需 `--rough`）

论文参数：未见过的 terrain heightfields，障碍物高度 [0, 15] cm。

实现方式：使用 `--rough` 标志，保留 flat policy 和 flat observation 空间，仅将地形替换为粗粝 terrain generator（7 种地形：pyramid_stairs、random_rough、wave_terrain 等）。Flat policy 看不到 `height_scan`，在粗粝地形上更容易摔倒，符合数据采集目的。

```bash
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/fall_trajs_rough \
    --num-trajs 5000 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/2026-06-11_01-03-02/model_1999.pt \
    --policy-task Mjlab-Velocity-Flat-Unitree-G1 \
    --rough \
    --device cuda:0
```

**不需要 rough policy**——policy 始终是 flat 的，只是地形换了。

## 5. 摔倒检测与保存条件

```python
# 每步检测
base_h = asset.data.root_link_pos_w[0, 2].item()
if base_h < 0.15 and step > 20:
    fell = True
    break  # episode 结束

# 保存条件：发生了摔倒 且 长度 ≥ 10 步
if fell and len(writer) >= 10:
    writer.save(f"traj_{n:06d}.pt")
```

**不保存的情况**：
- Robot 始终没有摔倒（policy 太强，扰动太弱）
- 轨迹太短 < 10 步（无意义的瞬态）
- episode 被 joint_vel_exceeded / time_out 终止但没摔倒

这意味着实际运行的 episode 数通常远大于 `--num-trajs`。

## 6. 轨迹文件格式

每条轨迹一个独立 `.pt` 文件：

```python
# 读取
d = torch.load("traj_000001.pt", map_location="cpu", weights_only=False)

# 结构
{
    "observations": torch.Tensor,  # [T, 63] float32
    "labels":       torch.Tensor,  # [T]    int64
    "T":            int,           # 轨迹总步数
}

# 标签含义
#   0  = safe      — t ≤ 2T/3，策略仍在正常控制
#  -1  = ambiguous — 2T/3 < t ≤ T-5步，过渡区（训练时 mask）
#   1  = falling   — t > T-5步，不可避免的摔倒最后 100ms
```

时序分割逻辑 (`dataset.compute_labels`)：

```
timestep:  0 ─────────── t1=2T/3 ─────── t2=T-5 ──── T (落地)
           │                │               │          │
label:     SAFE (0)        AMBIGUOUS (-1)  FALLING (1)
trained:   ✓               ✗ (masked)      ✓
```

**为什么 mask 掉 ambiguous 段？**论文指出 fall 的 transition 是 gradual 的，不存在一个明确的 "fall/no-fall" 分界点。mask 掉这段不会让模型学到错误的 label。

## 7. 数据规模参考

| 场景 | --num-trajs | --random | 大约耗时 | 数据量 | 用途 |
|------|------------|----------|---------|--------|------|
| 快速测试 | 500 | ✓ | 2-3 min (cpu) | ~7 MB | 验证流水线 |
| 小规模训练 | 5,000 | ✗ | ~1.5 h (cuda) | ~75 MB | 测试训练收敛 |
| 论文规模 | 81,920 | ✗ | ~23 h (cuda) | ~1.2 GB | 正式训练 |

## 8. 典型问题

### 编译慢

首次 `--device cuda:0` 运行会触发 MuJoCo-Warp JIT 编译（~30s-1min），输出类似：

```
Module _efc_contact_jac_sparse ... took 142.58 ms  (compiled)
Module mujoco_warp._src.ray ... took 774.93 ms    (compiled)
```

这是正常现象。编译结果缓存在 `~/.cache/warp/`，之后再运行只需微秒级加载 `(cached)`。

### 采集速率

影响采集速率的因素：
- **摔倒率**：policy 越强（训练步数多），机器人越不容易摔，每个有效轨迹需要更多 episode 尝试
- **物理步数**：速度跟踪 policy 在平地上可能跑几百步都不摔，需要通过调大扰动参数或缩短步数来提高效率
- **Warp kernel 编译**：仅首次运行，不影响后续

### 轨迹全部被拒绝

可能原因：
1. `max_steps` 太小，episode 结束时还没摔
2. Policy 太强，摔倒率极低 → 用 `--random` 模式测试，或增大扰动强度
3. base_height 阈值设置不当 → 当前为 0.15 m，适合 G1

## 9. 代码参考

| 文件 | 内容 |
|------|------|
| `collect_data.py` | 主采集脚本 + `collect_trajectories()` 函数 |
| `model.py` | `_extract_from_entity()` — 原始传感器数据提取 |
| `dataset.py` | `TrajectoryWriter`, `compute_labels()`, `load_trajectory()` |
| `train_predictor.py` | 使用采集数据训练 GRU predictor |
| `deploy.py` | `FallPredictorWrapper` — 部署时使用训练好的 predictor |
