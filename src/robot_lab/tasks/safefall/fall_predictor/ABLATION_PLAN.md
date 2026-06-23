# Warmup & Hysteresis Ablation Plan

验证 `FallPredictorWrapper` 中 `warmup_steps` 和 `confirmation_steps`（滞回）
两个机制是否有实际收益。

---

## 1. 背景

`deploy.py` 中 `FallPredictorWrapper.update()` 在原始 GRU 预测结果之上加了
两层后处理：

```python
# A. warmup — 前 N 帧强制返回 safe
if self._step_count <= self.warmup_steps:
    return False, self._last_prob

# B. hysteresis — 连续 M 帧 prob > threshold 才触发
if self._last_prob > self.threshold:
    self._consecutive_falling += 1
else:
    self._consecutive_falling = max(0, self._consecutive_falling - 1)

if self._consecutive_falling >= self.confirmation_steps:
    self._is_falling = True
```

论文未提及这两个机制。本实验验证它们对**预测时序精度**和**误报率 (FAR)**
的定量影响。

---

## 2. 实验指标

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **FAR**（False Alarm Rate） | safe 帧被误报为 falling 的比例 | `FP_safe / total_safe_frames` |
| **Detection Delay** | 从真实 falling 标签起点到 predictor 首次触发的时间 | `t_trigger − t2`（步，负数=提前，正数=滞后） |
| **Lead Time** | predictor 触发到 ground impact 的时间 | `(T − t_trigger) × 0.02 s` |
| **Miss Rate** | falling 帧中未被检测到的比例 | `1 − (TP_falling / total_falling_frames)` |

核心权衡：**FAR 越低越好（不要误报），Lead Time 越长越好（提前发现）**。
warmup 和 hysteresis 通过牺牲 Lead Time 来降低 FAR。本实验量化这个 trade-off。

---

## 3. 实验设计

### 3.1 数据准备

1. 采集 **500–1000 条**摔倒轨迹（见 `DATA_COLLECTION.md`）
2. 按 80/20 划分训练/验证集
3. 训练一个 fall predictor（`train_predictor.py`）
4. 另采集 **100 条纯正常行走轨迹**（不注入任何扰动、不用 `--rough`）作为
   **safe-only 测试集**

### 3.2 实验矩阵

对 `FallPredictorWrapper` 的三个参数做 full factorial 扫描：

| 参数 | 值 |
|------|----|
| warmup_steps | 0, 5, 10, 20, 40 |
| confirmation_steps | 1 (直通), 2, 3, 5, 8 |
| threshold | 0.3, 0.5, 0.7, 0.9 |

共 5×5×4 = **100 个配置组合**。

### 3.3 实验流程

对每个组合 `(warmup_steps, confirmation_steps, threshold)`：

```
1. 加载 predictor checkpoint
2. wrapper = FallPredictorWrapper(model, threshold, warmup_steps, confirmation_steps)

3. 对每条摔倒轨迹 (T 步):
   for t in 0..T-1:
       is_falling, prob = wrapper.update(obs[t])
       记录 t_trigger = 首次 is_falling==True 的时刻
   统计: Detection Delay、Lead Time、Miss Rate

4. 对每条 pure‑safe 轨迹:
   for t in 0..T-1:
       is_falling, prob = wrapper.update(obs[t])
       统计: FAR（safe 帧中 is_falling==True 的比例）

5. 汇总所有轨迹的结果
```

### 3.4 可视化

对实验结果做以下分析图（用 matplotlib）：

**图 1 — Trade-off 散点图**
x=FAR, y=Lead Time(均值) 的散点图，每个点是一个配置组合。
颜色按阈值变化，形状按 warmup_steps/confirmation_steps 变化。
理想的配置在左上角（高 Lead Time，低 FAR）。

**图 2 — Warmup 单变量分析**
固定 threshold=0.5, confirmation_steps=3 时，
x=warmup_steps, y1=FAR, y2=Lead Time 双纵轴折线图。

**图 3 — Hysteresis 单变量分析**
固定 threshold=0.5, warmup_steps=10 时，
x=confirmation_steps, y1=FAR, y2=Lead Time 双纵轴折线图。

**图 4 — 累积概率分布**
选最优 3 个配置的 Detection Delay 累积分布曲线——比较 delay 的集中程度。

---

## 4. 假设与判定标准

### H0-warmup: warmup_steps=0 和 warmup_steps>0 的 FAR 无差异

- 如果 `warmup_steps=0` 的 FAR 与 `warmup_steps=5` 接近（< 0.1% 差异）
  → warmup 无作用，建议去掉
- 如果 `warmup_steps=0` 的 FAR 显著高于 `warmup_steps>=5`
  → warmup 有作用，保留当前 10 或根据 trade-off 调整

### H0-hysteresis: confirmation_steps=1 和 confirmation_steps>1 的 FAR 无差异

- 如果 `confirmation_steps=1`（直通）的 FAR 已 < 0.1%（论文水平）
  → hysteresis 无作用，建议去掉
- 如果 `confirmation_steps=1` 的 FAR 显著高
  → hysteresis 有作用，选择最小的使 FAR < 0.1% 的 confirmation 值

### 预期结果

- **Warmup 很可能无作用**：如果 GRU 从零隐藏状态到一个
  reasonable 状态只需不到 200ms（10 步），那 warmup_steps=0 和 10 差异极小。
  warmup 真正的作用是防止从零初始状态的前几次 forward 误报——但前几步
  GRU 倾向于预测 near-uniform 概率（约 0.5），除非 threshold 很低否则不会触发。
  
- **Hysteresis 很可能有必要**：单帧尖峰是一个真实的问题——policy 突然的剧烈动作
  （跳跃、快速转身）或传感器噪声尖峰可能造成单帧 prob 跳到 0.8 以上。
  2-3 帧滞回可以在几乎不增加 Detection Delay（3 帧 = 60ms）的前提下大幅抑制 FAR。

---

## 5. 实现要点

### 5.1 纯正常行走轨迹采集

不需要额外写脚本——直接在 `collect_data.py` 基础上加 `--no-perturbations --no-terrain`
模式（不注入扰动、平地、不用 `--rough`）。每条 trajectory 全 label=0（safe-only）。

```bash
python -m robot_lab.tasks.safefall.fall_predictor.collect_data \
    --output data/safe_trajs \
    --num-trajs 100 \
    --policy-checkpoint logs/rsl_rl/g1_velocity/.../model_1999.pt \
    --device cuda:0
```

代码改动：`collect_data.py` 里加一个 `--no-perturb` flag，设 `PerturbConfig(all defaults)`。

### 5.2 实验脚本结构

```python
# ablation.py — proposed structure
def run_ablation(predictor_ckpt, fall_trajs_dir, safe_trajs_dir):
    model = FallPredictor(); model.load(predictor_ckpt); model.eval()

    results = []
    for warmup in [0, 5, 10, 20, 40]:
        for conf in [1, 2, 3, 5, 8]:
            for thresh in [0.3, 0.5, 0.7, 0.9]:
                wrapper = FallPredictorWrapper(model, thresh, warmup, conf)
                far, lead_time, miss_rate, delay = evaluate(wrapper, ...)
                results.append({...})

    # Plot & store results
```

### 5.3 使用现有的 evaluate 基础设施

- 帧级预测指标直接用 `dataset.compute_metrics` + `dataset.false_alarm_rate`
- 时序指标（lead time、detection delay）在序列级评估，需要额外的
  per-trajectory 循环追踪 `t_trigger`

---

## 6. 预期工作量

| 任务 | 预估时间 |
|------|---------|
| collect_data.py 加 `--no-perturb` flag | 15 min |
| 采集 100 条 pure‑safe 轨迹 | 30 min |
| 采集 1000 条 fall 轨迹 | 2–3 h (GPU) |
| 训练 predictor | 30 s |
| 编写 ablation.py | 1 h |
| 运行 100 配置 × ~1100 轨迹的评估 | 5–10 min (CPU) |
| 绘图 + 写结论 | 30 min |

---

## 7. 决策规则

```
if FAR(warmup=0, conf=1, threshold=0.5) < 0.1%:
    去掉 warmup 和 hysteresis，用最简单的单帧直通
elif FAR(conf=1) 显著 > FAR(conf>=2):
    保留 hysteresis，选择最小的 conf 使 FAR < 0.1%
    检查 warmup 是否需要（同上逻辑）
else:
    保留两者，选择最小参数满足 FAR 阈值的配置
```

输出：一份简短的分析报告，包含 4 张图 + 推荐的最优配置。
