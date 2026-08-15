# AdvancedChapterRL — 模块速览（公式 / 流程 / 文本）

本文件汇总 AdvancedChapterRL（强化学习进阶）目录相关 `.cpp` 的功能总结。每个模块以三种表达方式呈现：公式（关键计算）、流程（步骤化伪码）与文本说明，方便快速理解实现要点与数据流。

包含模块：

- AdvancedChapterRL.cpp（入口）
- BaseAdvanced.*（训练/测试框架）
- DuelingDQN.*（Dueling DQN）
- TRPO.*（Actor-Critic / 优势估计）
- Reinforce.*（REINFORCE 策略梯度）

---

## 1 AdvancedChapterRL.cpp（入口）

文本：

- 程序入口；构造并运行各算法示例（例如 TRPO、DuelingDQN、Reinforce 等）。

流程：

1. 构造算法对象（例如 `TRPO trpo;`）。
2. 调用运行方法（如 `trpo.PlayCartPole(1000);`）。
3. 等待用户输入结束。

公式：

- 无（控制/调度模块）

---

## 2 BaseAdvanced（训练 / 数据生成基类）

文本：

- 统一环境交互、训练数据生成与评估流程。子类需实现 `TakeAction`、`TrainGenerateItem1`、`TrainGenerateItem2`。

流程（伪码）：

1. PlayCartPole(max):
   - 调用 `GenerateTrainData(max)`，随后 `TestData(max)`。
2. GenerateTrainData(max):
   - 对每个 episode：
     - s = env.reset()
     - while not done and rewardCount < 上限:
       - a = TakeAction(s)
       - s1,r,done = env.step(a)
       - vList.push({s,a,r,s1,done})
       - TrainGenerateItem1(vList.back())
     - TrainGenerateItem2(vList)
     - 若 m_bEndGenerateTrain 则退出
3. TestData:
   - 多次使用 `TakeAction(s, true)` 执行评估并打印回报。

公式：

- 无统一损失；训练细节由子类实现（DQN / Policy Gradient 等）。

---

## 3 DuelingDQN（Dueling 架构的 DQN）

文本：

- 使用 Advantage 与 Value 分支合成 Q 值；经验回放 + 目标网络（periodic hard sync）+ MSE 损失。
- 关键函数：`CreateOptimizer`、`SyncTargetNet`、`TakeAction`、`Update`、`GenerateTrainData`。

公式（关键计算）：

- Q 合成：Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
- 目标值：q_target = r + γ * max_{a'} Q_target(s', a') * (1 - done)
- 损失：L = Mean( (Q(s,a) - q_target)^2 )

流程（训练步骤）：

1. 初始化 m_Qnet 与 m_TargetQnet，移动到设备。
2. 创建 Adam 优化器。
3. 初次同步目标网络（硬拷贝）。
4. 环境交互保存样本到回放缓冲；当缓冲满足最小样本量则采样批次训练。
5. Update():
   - samples = replay.sample(batchsize)
   - [s0,a,r,s1,done] = 转张量
   - q = Q_net(s0).gather(1,a)
   - q1 = max_a' Q_target(s1,a') → q1.view(-1,1)
   - q_target = r + γ * q1 * (1 - done)
   - loss = MSE(q, q_target); backward + step
6. 每 N 步调用 SyncTargetNet（当前实现为保存/加载临时文件）。

注意：

- 建议改为内存级别的 state_dict 拷贝以减少磁盘 I/O。
- 评估模式使用 NoGradGuard 并 argmax；训练模式使用 epsilon-greedy。

---

## 4 TRPO（Actor-Critic / 优势估计）

文本：

- 包含 Actor（PolicyNet）与 Critic（ValueNet）。主要做法：采集轨迹 → 计算 TD / GAE 优势 → 更新 Critic（MSE）并准备策略更新（TRPO 的 KL 约束在片段中未完整实现）。

公式（优势估计）：

- TD error:
  δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
- GAE（从末端反向累加）：
  A_t = ∑_{l=0}^{T-t-1} (γ λ)^l δ_{t+l}
  （代码实现：adv = γ λ * adv + δ，逆序累加）

流程：

1. 初始化 Actor 与 Critic，创建 Critic 优化器。
2. 通过 BaseAdvanced 采集轨迹 vList。
3. 在 TrainGenerateItem2:
   - v0 = Critic(s0)
   - v1 = r + γ * Critic(s1) * (1 - done)
   - td = v1 - v0
   - advantage = ComputeAdvantage(γ, λ, td)
   - 更新 Critic: criticLoss = mean(MSE(v0, v1.detach())); adam.step()
4. 策略/TRPO 更新（KL 约束）需在更完整实现中补充。

注意：

- ComputeAdvantage 将 td 拉到 CPU 做反向循环以保证数值正确并支持多列并行环境。

---

## 5 Reinforce（蒙特卡洛策略梯度）

文本：

- 基本 REINFORCE：收集整条 episode 的回报 G_t，按 −log π(a_t|s_t)*G_t 反向更新策略网络。

公式：

- G_t = ∑_{k=0}^{T-t} γ^k r_{t+k}
- 损失：L = - ∑_t log π(a_t|s_t) * G_t

流程：

1. 初始化 PolicyNet 与 Adam 优化器。
2. 每个 episode 采集 vList。
3. 反向遍历 vList：
   - G = γ * G + r_t
   - loss_increment = - log π(a_t|s_t) * G
   - 累积 backward()
4. adam.step() 更新参数。

注意：

- 可加入 baseline (value) 以降低方差（当前实现为原始 REINFORCE）。

---

## 统一要点（共性）

公式/概念共性：

- 折扣因子 γ：影响 q_target / G / advantage 的时间折扣。
- MSE：用于 Critic 或 Q 网络的回归目标。
- −logπ * G：用于策略梯度的目标。

通用训练流程：

1. 初始化网络与优化器，移动到 m_device（CPU/CUDA）。
2. 通过环境交互采集样本或轨迹（BaseAdvanced 驱动）。
3. 转换数据为张量（QwListToTensor / VectorDoubleTensor）。
4. 计算目标（q_target / advantage / G）。
5. 反向传播并优化参数（Adam）。
6. 同步或更新目标（如 target network 同步）。
7. 定期评估（TestData）。

---

## 工程级建议

- 目标网络同步：优先使用内存拷贝（state_dict）替代磁盘序列化以提升性能。
- 设备一致性：确保所有张量和模型在同一 device 上，避免隐式拷贝。
- 日志与检查点：添加训练日志（loss、reward 曲线）与 checkpoint 功能便于恢复与调参。
- 单元测试：对优势计算、目标计算等数值模块添加测试以保证正确性。

---
