# HoST Ground 在 MJLab 中的复现与排障方法论

## 1. 文档目标与复现边界

本文复盘如何将 HoST 的 Ground 任务从论文/Isaac Gym 语义迁移到 MJLab，并把“机器人站不起来”逐步定位为一组可验证的工程问题。重点不是记录一组偶然有效的超参数，而是说明：

1. 面对异常现象时先测量什么；
2. 如何建立候选原因并逐一排除；
3. 如何证明修正确实解决了原问题；
4. 如何避免训练指标很好、实际部署却失败。

当前完成范围仅为 Unitree G1 从 Ground 仰卧姿态起身：

```python
init_pos = (0.0, 0.0, 0.5)
init_quat = (0.70710678, 0.0, -0.70710678, 0.0)
```

最终策略在 `0 N` 牵引力和 `beta=0.25` 的部署条件下，独立评估 1024 回合：

```text
success_rate=1.0000
mean_max_base_height=0.7566
mean_max_head_height=0.7601
mean_best_upright_time=8.3929 s
```

最终模型：

```text
logs/rsl_rl/g1_host_ground/2026-07-28_22-54-13_supine_v3_latched/model_799.pt
```

本实现不是论文全部能力的等价复现。论文使用多 critic 和 L2C2，并覆盖多种地形/姿态；当前 MJLab 实现使用单 critic，针对奖励尺度做了重新平衡，并只验证 Ground 仰卧起身。G1 MJCF 也缺少论文中部分脚踝 keypoint，因此不能机械照搬所有奖励。

## 2. 总体方法：建立证据链，而不是盲目调参

每次排障统一使用下面的闭环：

```text
观察现象
  -> 把现象转换成可测指标
  -> 列出最可能破坏任务契约的原因
  -> 用最小实验隔离变量
  -> 修改一项语义或尺度
  -> 做短训练/静态检查
  -> 用独立评估和可视化确认
```

### 2.1 先核对“任务契约”

跨仿真器迁移时，名称相同不代表语义相同。开始长时间训练前，应逐项核对：

- 初始位姿的坐标系和四元数顺序；
- 被策略控制的关节集合及顺序；
- 动作是绝对位置、相对默认位置，还是相对当前位置；
- 动作裁剪和缩放发生在什么位置；
- PD target 到底是什么；
- reset API 的 range 表示绝对值、乘法比例还是加法偏移；
- reward 使用世界坐标、机体坐标还是相对坐标；
- 论文中的 body/site/keypoint 在当前资产中是否真实存在；
- curriculum 的成功条件在当前资产上是否可达；
- curriculum 状态是否随 checkpoint 保存；
- train、evaluate、play 是否使用相同的动作和环境参数。

### 2.2 三层验证标准

仅看训练 reward 不足以证明成功。这里采用三层验证：

1. **静态/单步验证**：动作维度、reset 范围、buffer 初值、reward 是否为常数。
2. **批量数值验证**：固定 checkpoint，在明确的 `force/beta` 下运行 256 或 1024 回合。
3. **部署链路验证**：使用标准 `play` 录制完整回合，并抽查起身前、起身中和回合末尾画面。

只有三层同时通过，才认为任务完成。

## 3. 异常现象与逐项排查

### 3.1 初始姿态看起来有歧义

**现象**

仅凭相机视角难以判断机器人是仰卧还是俯卧，四元数在不同引擎中的约定也容易混淆。如果初态方向错误，后续所有奖励和动作都会在错误任务上优化。

**分析方法**

不要根据四元数符号或单一视角猜测。创建单环境 RGB 渲染，在策略动作前后保存初始帧，同时检查胸部模型朝向。最终由实际画面和任务定义确认 Ground 为胸部朝上的仰卧姿态。

**修正**

固定使用：

```python
(w, x, y, z) = (0.70710678, 0.0, -0.70710678, 0.0)
```

**经验**

位姿问题必须通过渲染或坐标轴数值验证，不能用“看起来像”代替证据。用户或数据集对姿态的明确语义优先于从模型外观做二次推断。

### 3.2 动作空间错误地包含了 29 个 DoF

**现象**

原实现用 `actuator_names=(".*",)` 控制全部 29 个执行器，而论文明确使用 23-DoF G1。策略输出、历史动作、关节观测和 reward 因而建立在错误的控制空间上。

**排查**

1. 从论文确认 action dimension 为执行器数量，实验使用 23-DoF G1。
2. 打印 MJLab `ActionManager` 的 active terms 和 dimension。
3. 对照 G1 joint names，明确论文策略控制的腿、腰 yaw、肩、肘和 wrist roll。

**修正**

在 `env_cfgs.py` 中定义 `_POLICY_JOINT_NAMES`，让 action、joint position/velocity observation、reset 和 tracking reward 使用同一关节集合。修正后环境输出：

```text
Active Action Terms (shape: 23)
actor observation shape: 456
critic observation shape: 474
```

**经验**

动作维度是最先应该 assert 的接口契约。维度能运行并不代表维度正确；错误关节集合会让网络容量、观测历史和奖励全部错位。

### 3.3 beta 课程存在，但没有真正作用于动作

**现象**

配置和 observation 中出现了 `beta`，但实际 PD target 没有乘它。这样课程日志即使显示 beta 在下降，机器人的控制权限也没有变化。

**论文契约**

论文定义：

```text
pd_target = current_joint_pos + beta * clamp(action, -1, 1)
```

beta 限制每个策略周期的最大关节目标变化，目的是逐步限制动作速度和冲击，而不是普通 reward 权重。

**排查**

沿动作数据流检查：policy output -> clip -> action term -> processed action -> PD target。不能只看 cfg 中是否有 beta。

**修正**

`mdp/actions.py` 现在执行：

```python
actions = actions.clamp(-1.0, 1.0)
actions = actions * active * host_action_rescale
target = current_joint_pos + actions
```

训练从 `beta=1.0` 开始，每次成功减 `0.02`，下限为 `0.25`。

**验证**

强制将同一个早期 checkpoint 置于不同 beta：

- `model_400.pt` 在训练当时约 `beta=0.60` 可以起身；
- 强制 `beta=0.25` 时成功率为 `0%`，最大骨盆高度仍约 `0.50 m`。

这证明 beta 已经真实改变控制难度，而不是只改变日志。

### 3.4 joint tracking reward 跟踪了错误目标

**现象**

原 reward 使用：

```python
target = default_joint_pos + raw_action
```

但实际控制是相对当前关节位置，并且还包含 clip、无控制阶段 gating 和 beta。reward 惩罚的目标与 PD 真正收到的目标不同。

**排查**

对任何 tracking reward 追问两个问题：

1. reward 中的 target 是否与 actuator 最终 target 是同一个 tensor/语义？
2. 是否遗漏了 action processing 中的 clip、scale、offset 或 delay？

**修正**

动作 term 暴露 `target_joint_pos`，reward 直接读取 processed PD target，并只比较策略控制的 23 个关节。

**经验**

控制目标应只有一个事实来源。重新根据 raw action 推导 target，极易与实际 action pipeline 漂移。

### 3.5 reset API 语义误解导致关节严重越界

**现象**

初始短训练中关节限位 penalty 约为 `-18.72`，机器人动作异常且优化几乎被惩罚项支配。

**根因**

原代码调用 MJLab 的 `reset_joints_by_offset(position_range=(0.5, 1.5))`。这里的 range 是给每个默认关节角增加 `0.5~1.5 rad`，不是论文想要的 `0.5~1.5` 倍缩放。大量关节在回合开始时就接近或越过限位。

**排查**

1. 直接阅读被调用 reset 函数的实现，而不是根据函数名猜测。
2. reset 后统计每个关节到 soft limit 的距离。
3. 分项观察 `regu_dof_pos_limits`，判断异常是在策略动作后产生，还是 reset 时已经存在。

**修正**

实现论文式 reset：

```text
joint_pos = default_joint_pos * U(0.9, 1.1) + U(-0.05, 0.05)
joint_pos = clip(joint_pos, soft_lower, soft_upper)
```

修正后早期限位 penalty 从约 `-18.72` 降至约 `-1.25`，最终训练末段约为 `-0.50`。

**经验**

仿真迁移中，range 参数是高风险点。必须确认其单位和运算方式，并在训练前采样 reset 分布。

### 3.6 牵引力初值与论文不一致，探索能力不足

**现象**

没有牵引力时，早期策略始终无法从地面进入跪坐/起身阶段；原实现的初始牵引力只有 `100 N`。

**排查**

论文消融已经表明 Ground 无 force curriculum 时成功率为 0。Appendix B 明确给出：

```text
initial force = 200 N
decrement = 20 N
lower bound = 0 N
```

**修正**

训练模式改为从 `200 N` 开始，身体接近竖直后才向 pelvis 施加世界坐标 `+Z` 方向的力；成功后逐级下降到 `0 N`。

**验证**

第一阶段 `model_100.pt`：

- 默认辅助条件下成功率约 `91.02%`；
- 同一 checkpoint 强制 `0 N` 时成功率为 `0%`。

这不是策略最终成功，而是证明牵引力课程确实承担了“先探索到可行运动”的作用。直接去掉外力会破坏学习过程。

### 3.7 奖励函数在跨资产迁移后含义发生变化

#### 3.7.1 orientation reward 被高度 gate 截断

原 orientation reward 只有骨盆高度超过 `0.45 m` 才生效，导致最需要“把身体转正”的早期阶段缺少方向梯度。修正为全阶段稠密方向奖励。

#### 3.7.2 head height 使用了错误参考系

原实现计算 `torso_z - feet_z`，但论文 task reward 使用世界坐标中的 head height。改成 torso/head 的世界 `z` 高度。参考系错误会让躺姿、坐姿和站姿之间的 reward 排序失真。

#### 3.7.3 target base height 的核函数不一致

由 `exp(-20 * abs(error))` 改为论文式高斯形态 `exp(-20 * error^2)`，使目标附近的梯度更平滑。

#### 3.7.4 ground-parallel reward 是恒定伪奖励

论文基于每只脚多个 ankle keypoint 的高度方差评估脚掌是否平行地面。当前 MJCF 每侧只有一个可用 ankle body；单点方差恒为 0，reward 恒为正，完全不能区分动作好坏。

处理方式不是“调低权重”，而是关闭该项：

```text
style_ground_parallel weight = 0
```

如果将来补充多个可靠 foot site，再恢复该奖励。

#### 3.7.5 单 critic 下奖励尺度失衡

论文用多 critic 分组优化 task/style/regularization/post-task reward；当前 runner 是单 critic。机械照搬权重后，关节限位、速度等 penalty 会吞没起身信号。

处理方法：先记录每个 reward term 的实际 episode magnitude，再对数量级做平衡，而不是只比较配置 weight。正则项整体降低，真实 joint tracking error 的权重相应提高。

**经验**

奖励迁移必须检查三件事：物理量、参考系、数值分布。函数名相同不能证明 reward 等价；恒定 reward 应直接删除或关闭。

### 3.8 curriculum 成功条件不可达或只在错误时刻检查

**现象**

机器人某一时刻已经站起，但课程几乎不推进；或者在回合末稍微下沉后，被判定整回合失败。

**根因一：资产映射错误**

论文用真实 head height 达标作为课程条件。MJCF 没有对应的独立 head body，使用 `torso_link - feet` 再套用论文 head threshold，使条件在当前几何定义下很难达到。

**根因二：只看回合最后一帧**

起身是动态任务。只检查 reset 前最后一帧，会丢掉“中途已经完成起身”的信息，导致课程推进极慢。

**修正**

1. 用当前资产可解释的条件定义阶段完成：

   ```text
   base_height > 0.65 m
   projected_gravity_z < -0.8
   ```

2. 每步更新 `host_standup_success` latch；一旦满足，本回合保持为 true，reset 时课程读取 latch。
3. 每回合 reset latch，避免成功状态泄漏到下一回合。

**训练证据**

第二阶段使用旧的末帧条件，约 iteration 250 时：

```text
mean_force ~= 159 N
mean_action_rescale ~= 0.959
success_rate ~= 0.14
```

加入 latch 后，外力逐步降至 `0 N`，课程成功触发率保持 `1.0`，beta 再继续从约 `0.74` 下降到 `0.25`。

**经验**

课程条件服务于“是否允许提高难度”，最终评估条件服务于“是否稳定完成任务”，两者可以不同。课程可以 latch 一次到达，最终验收则必须要求持续站立。

### 3.9 训练探索噪声在后期过强

**现象**

策略已经进入可行区间后仍有较大动作方差，起身和稳定站立的行为不够集中。

**处理**

将 PPO entropy coefficient 从 `0.01` 降至 `0.001`，继续保留探索但减少后期随机性。这项改动与 success latch 一起用于第三阶段续训。

**注意**

这不是单独可证明的唯一根因，因此不应声称“降低 entropy 就解决了起身”。它是课程已能推进后的稳定化手段，主要证据仍来自独立 checkpoint 对比。

### 3.10 checkpoint 不保存课程 buffer，加载后难度会倒退

**现象**

加载一个已经训练到 `force=0、beta=0.25` 的 checkpoint 后，环境可能重新从 `force=200、beta=1.0` 初始化。策略网络参数保存了，但环境上的 `host_traction_force` 和 `host_action_rescale` 不在 checkpoint 中。

**排查**

不要根据 checkpoint 所在训练轮次推断部署参数。环境创建后直接打印：

```python
print(env.host_traction_force)
print(env.host_action_rescale)
```

**修正和验证原则**

- 续训时明确知道课程 buffer 会重新初始化，不能把 reload 当作完全恢复环境状态；
- 独立评估显式设置 `force=0` 和 `beta=0.25`；
- play 配置使用最终部署初值，而训练配置仍使用课程初值。

**经验**

RL checkpoint 通常只保证恢复网络/优化器，并不自动保存 Manager 环境的任意自定义 tensor。所有影响策略输入或动作的环境状态都必须单独审计。

### 3.11 “成功率很高”但机器人站不稳

**现象**

`model_600.pt` 在 `0 N、beta=0.25` 下 1024 回合短时起身成功率已经接近 100%，但单回合 trace 显示机器人只站住约 0.88~1.15 秒，随后逐渐下沉。这还不满足论文“达到目标高度并保持到回合结束”的成功含义。

**排查**

二值 success 只回答“是否曾连续站立 0.5 秒”，无法回答“之后是否倒下”。因此增加并比较：

- 最大骨盆高度；
- 最大 torso 相对脚高度；
- 最长连续直立时间；
- 回合中多个固定 step 的 `base_z` 和 `gravity_z`；
- 完整 10 秒回放。

**checkpoint 对比**

```text
model_600: success ~= 99.9%, mean best upright time ~= 1.15 s
model_700: success = 100%,   mean best upright time ~= 1.30 s
model_799: success = 100%,   mean best upright time ~= 8.39 s
```

**修正**

课程到达 `force=0、beta=0.25` 后没有立即停止，而是在固定最终难度上继续训练约 220 iterations。最终 post-task height/orientation/velocity reward 明显增长，策略学会的不再只是“碰到站立阈值”，而是维持站立。

**经验**

成功率必须和持续时间、末帧状态、视频一起看。对于阶段性任务，“曾经成功”和“稳定完成”是不同指标。

### 3.12 独立评估成功，但标准 play 先起身再倒下

**现象**

独立评估显式设 `beta=0.25` 时稳定站立；最初的标准 play 录像却在起身后倒下。说明问题不在模型本身，而在 evaluate 与 play 的环境初始化差异。

**隔离变量**

1. 同一个 `model_799.pt`；
2. 同一个 Ground play env cfg；
3. evaluate 显式写入 beta，标准 play 依赖默认初始化；
4. 直接读取 play 环境创建前后 buffer。

探针发现：

```text
play 创建后实际 beta=1.0
```

**根因**

这是初始化顺序问题，不是配置值写错：

1. observation 构造时最先发现 beta buffer 不存在，于是创建为 `1.0`；
2. reset event 只在“不存在”时初始化，因此不能覆盖成 play 的 `0.25`；
3. curriculum 在发现 force buffer 不存在时，又曾无条件把 beta 写成 `1.0`。

**修正**

1. 将 `initial_action_rescale` 作为单一配置值传给 observation、action、event、curriculum；
2. training 使用 `1.0`，play 使用 `0.25`；
3. curriculum 分别判断 force buffer 和 beta buffer 是否存在，不再捆绑初始化。

**修正后探针**

```text
play 环境创建后：force=None, beta=0.25
执行一步后：     force=0.0,  beta=0.25

train action beta=1.0
train observation beta=1.0
train force=200.0
train curriculum beta=1.0
```

重新用标准 play 录制后，1 秒处于坐起阶段，5 秒、8 秒和 9.5 秒均保持站立。

**经验**

配置对象里的值正确，不代表运行时 buffer 正确。存在多个 lazy initializer 时，必须验证“谁先创建、谁会覆盖”，并在环境创建后和第一步后分别读取实际值。

## 4. 训练阶段如何设计成可诊断实验

本次没有一次性跑满训练，而是将训练分成能回答具体问题的阶段。

### 4.1 Baseline 和 smoke test

```text
2026-07-28_22-24-29_baseline_bench
2026-07-28_22-39-58_fixed_smoke
```

只跑 1~2 iterations，用于确认：

- 环境能创建；
- action shape 为 23；
- 没有 NaN/异常 termination；
- reset 后限位 penalty 回到合理量级；
- reward term 能写入日志。

短测试通过前，不值得消耗时间进行长训练。

### 4.2 第一阶段：确认辅助力能打开探索通路

```text
2026-07-28_22-40-48_supine_v1/model_100.pt
```

默认辅助条件成功率约 `91%`，强制 `0 N` 成功率为 `0%`。结论是动作/reward 基本能产生起身行为，但策略仍依赖牵引力，下一步应修课程而不是直接宣布成功。

### 4.3 第二阶段：观察课程是否真实推进

```text
2026-07-28_22-48-03_supine_v2_curriculum/model_200.pt
```

跟踪三条课程指标：

```text
mean_force
mean_action_rescale
success_rate
```

课程推进过慢暴露了末帧判定问题，从而引出 success latch。

### 4.4 第三阶段：逐步达到最终部署难度

```text
2026-07-28_22-54-13_supine_v3_latched
```

训练过程中的关键点：

```text
iteration ~330: force=0 N, beta~0.74, curriculum success=1.0
iteration ~400: force=0 N, beta~0.60
iteration ~520: force=0 N, beta~0.36
iteration ~580: force=0 N, beta=0.25
iteration 580~799: 固定最终难度继续优化稳定站立
```

### 4.5 为什么要用“越级测试”

当训练还在 `beta~0.60` 时，提前把 `model_400` 强制放到 `beta=0.25`，结果为 0% 成功。这项失败很有价值：它证明课程参数确实生效，也说明不能因为训练 batch 成功率为 100% 就提前停止。

## 5. 日志应该怎么看

建议同时观察四组指标，而不是只看 `Mean reward`。

### 5.1 任务进度

```text
task_orientation
task_head_height
target_orientation
target_base_height
```

task reward 上升说明能进入起身过程；post-task reward 持续上升更能说明起身后站得住。

### 5.2 动作质量和数值健康

```text
regu_dof_pos_limits
regu_joint_tracking_error
regu_action_rate
regu_smoothness
regu_torques
```

限位 penalty 极大通常优先检查 reset/action semantics，而不是立刻改 PPO。tracking error 异常则应核对 reward target 与 PD target。

### 5.3 课程进度

```text
Curriculum/traction_force/mean_force
Curriculum/traction_force/mean_action_rescale
Curriculum/traction_force/success_rate
```

force/beta 长时间不变说明成功条件不可达或课程没有被调用；它们下降过快且成功率随后崩溃，说明课程步长过大。

### 5.4 稳定性

训练日志中的 batch success latch 只表示本回合曾达到站立条件。最终稳定性必须由独立评估的连续直立时间和完整视频判断。

## 6. 独立评估、截图和录像

### 6.1 为什么需要独立评估

训练 batch 同时受课程状态、探索噪声和并行 reset 影响。独立评估：

- 不更新网络；
- 只加载 actor 做 inference；
- 固定 seed；
- 明确覆盖 `force=0、beta=0.25`；
- 运行完整 10 秒回合；
- 聚合多回合结果。

评估实现位于 `evaluate.py`。成功条件是：

```text
base_height > 0.65 m
projected_gravity_z < -0.9
连续保持 >= 0.5 s
```

同时输出最长连续直立时间，避免短时跨过阈值被误认为稳定站立。

### 6.2 批量评估命令

```bash
env MPLCONFIGDIR=/tmp/mpl-host-eval \
    XDG_CACHE_HOME=/tmp/xdg-host-eval \
    CUDA_VISIBLE_DEVICES=0 \
    MUJOCO_GL=egl \
/data1/tanglong/miniconda3/envs/env_mjlab_e/bin/python \
    -m robot_lab.tasks.host.evaluate \
    --checkpoint logs/rsl_rl/g1_host_ground/2026-07-28_22-54-13_supine_v3_latched/model_799.pt \
    --num-envs 256 \
    --episodes 1024 \
    --device cuda:0 \
    --force 0 \
    --action-rescale 0.25
```

### 6.3 轨迹探针

```bash
/data1/tanglong/miniconda3/envs/env_mjlab_e/bin/python \
    -m robot_lab.tasks.host.evaluate \
    --checkpoint logs/rsl_rl/g1_host_ground/2026-07-28_22-54-13_supine_v3_latched/model_799.pt \
    --num-envs 1 \
    --episodes 1 \
    --force 0 \
    --action-rescale 0.25 \
    --trace
```

trace 在固定 step 输出 `base_z`、相对 torso 高度、`gravity_z`、force 和 beta，适合判断“从未起身”“短暂站起”“逐渐下沉”三种不同失败模式。

### 6.4 截图

环境使用 `render_mode="rgb_array"`，策略运行到指定 step 后调用 `base_env.render()`，再由 PIL 保存 PNG：

```bash
/data1/tanglong/miniconda3/envs/env_mjlab_e/bin/python \
    -m robot_lab.tasks.host.evaluate \
    --checkpoint logs/rsl_rl/g1_host_ground/2026-07-28_22-54-13_supine_v3_latched/model_799.pt \
    --num-envs 1 \
    --episodes 1 \
    --force 0 \
    --action-rescale 0.25 \
    --snapshot artifacts/host_supine_stable_stand.png \
    --snapshot-step 250 \
    --snapshot-only
```

策略频率为 50 Hz，step 250 约为第 5 秒。

### 6.5 标准 play 录像

```bash
env MPLCONFIGDIR=/tmp/mpl-host-play \
    XDG_CACHE_HOME=/tmp/xdg-host-play \
    CUDA_VISIBLE_DEVICES=0 \
    MUJOCO_GL=egl \
/data1/tanglong/miniconda3/envs/env_mjlab_e/bin/play \
    Mjlab-HoST-Ground-Unitree-G1 \
    --checkpoint-file logs/rsl_rl/g1_host_ground/2026-07-28_22-54-13_supine_v3_latched/model_799.pt \
    --num-envs 1 \
    --device cuda:0 \
    --video True \
    --video-length 500 \
    --video-width 640 \
    --video-height 480 \
    --viewer auto
```

输出视频：

```text
logs/rsl_rl/g1_host_ground/2026-07-28_22-54-13_supine_v3_latched/videos/play/rl-video-step-0.mp4
```

视频为 10 秒、500 帧、50 fps。注意 `play` 录像完成后 viewer 仍会运行，需要手动关闭；这不影响已经写入的视频。

## 7. 可复用排障清单

### 7.1 开始训练前

- [ ] 渲染并确认初始姿态；
- [ ] 打印 action dimension 和 joint names；
- [ ] 手算一个动作经过 clip/scale/offset 后的 PD target；
- [ ] reset 1000 次并统计 joint limit violation；
- [ ] 检查每个 reward 是否随状态变化，排除恒定项；
- [ ] 确认所有 reward 的坐标系、单位和 body/site 映射；
- [ ] 打印 force、beta 等自定义 buffer 的真实初值；
- [ ] 确认 train/play 配置差异是有意设计。

### 7.2 短训练后

- [ ] 每个 reward term 是否为有限值；
- [ ] penalty 是否比 task reward 大几个数量级；
- [ ] termination 是否异常频繁；
- [ ] force/beta 是否按预期推进；
- [ ] 同一 checkpoint 在有/无辅助条件下表现是否符合阶段预期。

### 7.3 宣布成功前

- [ ] 在最终 `force/beta` 下独立评估至少 256，最好 1024 回合；
- [ ] 不只报告 success rate，还报告连续直立时间；
- [ ] 检查回合末状态，不接受短暂越过阈值；
- [ ] 使用标准 play 而不只是自定义 evaluator；
- [ ] 抽查完整视频多个时间点；
- [ ] 环境创建后和第一步后读取部署 buffer；
- [ ] 运行语法和 diff 检查。

## 8. 本次复现最重要的结论

1. **先修语义，再调超参数。** 29/23 DoF、reset range 和 PD target 错误都不是 PPO 调参可以弥补的。
2. **失败的对照实验很有价值。** `model_400 + beta=0.25` 的 0% 和 `model_100 + force=0` 的 0%，分别证明了 action curriculum 和 force curriculum 确实在工作。
3. **资产映射必须重新定义可达指标。** 缺少 head body 或 ankle keypoint 时，应选择可解释的替代量或关闭奖励，不能保留形式相同但物理意义为空的代码。
4. **课程成功不等于最终成功。** latch 适合推动课程，但验收必须要求持续站立。
5. **checkpoint 不等于完整系统状态。** 网络、环境 buffer 和 play 默认值必须分别验证。
6. **最终证据必须跨越训练、评估和部署三条链路。** reward 上升、批量数值成功、标准 play 完整回放缺一不可。
