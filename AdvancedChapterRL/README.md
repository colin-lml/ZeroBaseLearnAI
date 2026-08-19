# AdvancedChapterRL — 代码文件速览与说明

本 README 补充说明当前目录下各个 .cpp 文件的功能要点、关键接口和已知注意事项，便于快速定位实现和调参。内容面向熟悉 C++/libtorch 的开发者，简洁明了。

总体说明
- 构建工具：CMake（Ninja），C++20。
- 依赖：libtorch（PyTorch C++ API）。
- 运行入口：AdvancedChapterRL.cpp -> main()，在 main 中按需构造并调用各算法的 Play / GenerateTrainData。
- 公共基类：BaseAdvanced.* 负责环境交互、轨迹采集与测试流程，子类实现 TakeAction、TrainGenerateItem1/2。

文件逐一说明（按 .cpp）

- AdvancedChapterRL.cpp
  - 程序入口，创建各算法对象（DeepQNetwork、DuelingDQN、PolicyGradient、ActorCritic、TRPO、PPO、DDPG）。
  - 在此切换/运行具体算法示例（通过注释启用不同算法的 Play 调用）。

- pch.cpp / pch.h
  - 预编译头，统一包含常用头文件、环境和模块头，简化编译与引用。

- CartPoleEnv.cpp / CartPoleEnv.h（环境）
  - CartPole 环境封装（reset / step / state/action 维度与上下界）。
  - 供各算法统一调用。

- BaseAdvanced.cpp / BaseAdvanced.h
  - 统一训练/评估框架：
    - Play(maxCount) 调用 GenerateTrainData(maxCount) -> TestData(maxCount)。
    - GenerateTrainData 负责按 episode 与 step 与 env 交互，收集 QwList 并调用子类 TrainGenerateItem1/2。
    - TestData 使用 TakeAction(..., true) 做评估运行。
  - 还包含：Categorical 分布封装、参数复制/向量化工具（CopyModuleParameters、ParametersToVector、VectorToParameters）。
  - 注意：Categorical 与参数工具要求 device 与 dtype 一致；调用端需保证。

- DeepQNetwork.cpp / DeepQNetwork.h
  - 标准 DQN 实现（Q 网络、目标网络、经验回放、MSE loss、epsilon-greedy）。
  - 要点：Q(s,a) gather、q_target = r + γ * max_a' Q_target(s',a') * (1-done)。
  - 建议：目标网络同步使用内存级 state_dict 拷贝以避免磁盘 I/O。

- DuelingDQN.cpp / DuelingDQN.h
  - Dueling 架构：Value 和 Advantage 分支合成 Q 值（Q = V + A - mean(A)）。
  - 其它逻辑与 DQN 类似（回放、目标网络、训练步骤）。

- PolicyGradient.cpp / PolicyGradient.h
  - REINFORCE（蒙特卡洛）实现：
    - 收集整条 episode，逆序累积回报 G，loss = -logπ * G。
  - 要点：方差大，建议加 baseline 或 advantage 标准化以稳定训练。

- ActorCritic.cpp / ActorCritic.h
  - 经典 Actor-Critic：
    - Critic 用 MSE 回归 v1 = r + γ V(s1)。
    - Actor 用 −logπ(a|s) * td.detach() 作为损失（td = v1 − v0）。
  - 要点：确保 actor/critic 梯度清零顺序与 device/dtype 一致。

- TRPO.cpp / TRPO.h
  - TRPO 风格实现要点：
    - Actor(PolicyNet) 与 Critic(ValueNet) 并行使用。
    - GAE 优势估计（ComputeAdvantage），Critic 用 MSE 更新。
    - 计算 KL、一阶梯度、Hessian-vector product、共轭梯度求解方向，再做 line-search（KL 约束）。
  - 已知风险与建议：
    - 数值稳定性敏感：建议对 advantage 做归一化、对 Hv 加阻尼 (damping)、增加 CG 迭代数并打印 KL/surrogate 以调试。
    - 确认 KL 的方向/实现是否与理论一致（old vs new），并在 line-search 中检测 NaN。

- PPO.cpp / PPO.h
  - PPO 实现要点：
    - 计算 td、GAE advantage、adv 标准化。
    - 多 epoch 优化 actor，使用 clip(ratio, 1-ε, 1+ε) 的 surrogate。
  - 已知问题与建议：
    - 计算 oldLogProbs 时需加小常数防止 log(0) 导致 NaN（例如 +1e-8）。
    - 保持 adv 的 detach/归一化与 device/dtype 一致。
    - 建议在训练时打印 adv mean/std、KL、actor loss 以便调参。

- DDPG.cpp / DDPG.h
  - 连续动作算法（actor = 连续策略网络，critic = Q(s,a)）。
  - 核心步骤：target networks、软更新（SoftUpdate）、critic MSE、actor 目标为 −E[Q(s, actor(s))]。
  - 已知问题与建议：
    - TakeAction 中用 noise 时需保证 noise 与 action shape 一致（使用 randn_like），对张量做 clamp，然后再取标量/向量返回，避免 shape mismatch。
    - SoftUpdate 采用参数逐元素更新实现正确，注意 m_tau 值与学习率匹配。

- PPO/其他通用实现细节
  - 优势估计（GAE）在多处复用，ComputeAdvantage 实现把 td 拉到 CPU 并做列级别反向累积；注意性能与 device 切换开销。
  - 参数向量化工具（ParametersToVector / VectorToParameters）用于 TRPO 的整体参数更新。

诊断与调参建议（简短）
- 统一 device 与 dtype（所有张量与模型尽量用 float32 同一 device）。
- 对 advantage 做均值-方差归一化（PPO/TRPO 已有实现，但需在所有策略更新处保持一致）。
- TRPO：增加 Hv 阻尼、增大 CG 迭代、打印 CG 残差/KL/step norm 有助定位不稳定。
- PPO：确保 oldLogProbs 使用 epsilon 避免 log(0)；多 epoch + clip 通常收敛更快。
- DDPG：用 randn_like 保证噪声 shape；对多维动作返回矢量接口或明确 squeeze 语义。
- DQN 系列：target sync 使用 state_dict 内存拷贝；检查 replay buffer 大小与采样策略。

如果需要，我可以：
- 直接将上述诊断与修改建议合并为 README.md（已生成）并提交；
- 或在代码中按建议做最小可回滚修补（PPO log epsilon、DDPG noise shape、TRPO 数值阻尼与 adv 归一化、可选日志打印）。
