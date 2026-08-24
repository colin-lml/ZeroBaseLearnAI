# AdvancedChapterRL — 代码文件速览与说明

本 README 补充说明当前目录下各个 .cpp 文件的功能要点、关键接口和已知注意事项，便于快速定位实现和调参。内容面向熟悉 C++/libtorch 的开发者，简洁明了。

## 项目概览

`AdvancedChapterRL` 是一个基于 **libtorch（PyTorch C++ API）** 实现的强化学习示例程序。它内置了 CartPole 离散动作环境与 Pendulum 连续动作环境，并提供 DQN、策略梯度、Actor-Critic、TRPO、PPO、DDPG 和 SAC 的最小训练/评估实现。

当前可执行程序名为 `AdvancedChapterRL`。入口在 `AdvancedChapterRL.cpp`；`main()` 中通过保留一个算法的 `Play(...)` 调用并注释其他调用来选择要运行的算法。当前默认执行：

```
int main()
{
    DeepQNetwork  deepQN;
    //deepQN.Play(400);
    //deepQN.DoubleDQN(400);
    DuelingDQN duelingDQN;
    //duelingDQN.Play(400);

    PolicyGradient policyGradient;
    //policyGradient.Play(1000);

    ActorCritic actorCritic;
    //actorCritic.Play(1000);

    TRPO trpo;
    //trpo.Play(500);

    PPO ppo;
    //ppo.Play(500);
    DDPG ddpg;
    //ddpg.Play(200);

    SAC sac;
    //sac.Play(200);

    cin.get();
    return 0;
}


```

## 依赖与构建

| 项目            | 当前配置                       |
| ------------- | -------------------------- |
| CMake 最低版本    | 3.8                        |
| 已使用的 CMake 版本 | 3.31.6-msvc6               |
| C++ 标准        | C++20                      |
| 生成器           | Ninja                      |
| 深度学习库         | libtorch / PyTorch C++ API |
| 平台相关头文件       | Windows（`windows.h`）       |

`CMakeLists.txt` 通过 `find_package(Torch REQUIRED ...)` 查找 libtorch，当前 `torchpath` 配置为 `D:/libtorch_gpu2.11.0/debug`。在本机构建前，应将该路径改为实际 libtorch 安装目录，或调整 CMake 配置以传入对应的 Torch 路径。

在 `AdvancedChapterRL` 目录执行：

cmake -S . -B build -G Ninja
cmake --build build

CMake 对 Debug 配置设置的运行时输出目录为 `AdvancedChapterRL/X64/`，其他配置使用 `bin2/`。运行时还必须保证 libtorch 所需的 DLL 可被系统找到。

## 运行模型

所有算法继承 `BaseAdvanced` 并使用相同的生命周期：

1. `Play(maxCount)` 固定随机种子为 `12`，执行训练后执行评估。
2. `GenerateTrainData(maxCount)` 与环境交互，生成 `(state, action, reward, nextState, done)` 转换。
3. 每个 episode 结束时调用 `TrainGenerateItem2(...)`；需要逐步更新的算法也会在每一步调用 `TrainGenerateItem1(...)`。
4. `TestData(...)` 以预测模式执行评估，单轮最多运行 500 步。

设备会在 `BaseAdvanced` 构造时自动选择 CUDA（可用时）或 CPU。`QwListToTensor(...)` 将批量转移为 `float32` 状态、奖励和终止标志；离散动作使用 `int64`，连续动作使用 `float32`。

## 环境

| 环境            | 状态                                      | 动作                   | 终止与奖励                       |
| ------------- | --------------------------------------- | -------------------- | --------------------------- |
| `CartPoleEnv` | 4 维：`x, x_dot, theta, theta_dot`        | 2 个离散动作              | 越界或杆角超过阈值终止；未终止时奖励为 `1`     |
| `PendulumEnv` | 3 维：`cos(theta), sin(theta), theta_dot` | 1 维连续力矩，范围 `[-2, 2]` | 200 步截断；奖励遵循 Pendulum-v1 形式 |

`BaseAdvanced` 默认创建 `CartPoleEnv`。`DDPG` 和 `SAC` 构造时传入 `false`，因此使用 `PendulumEnv`。目前两种连续动作算法均显式限制为一维动作空间。

## 算法实现

| 算法               | 环境/动作空间         | 实现摘要                                                                                  |
| ---------------- | --------------- | ------------------------------------------------------------------------------------- |
| `DeepQNetwork`   | CartPole / 离散   | 经验回放、epsilon-greedy、在线 Q 网络与目标 Q 网络；`DoubleDQN(...)` 启用在线网络选动作、目标网络估值的 Double DQN 目标。 |
| `DuelingDQN`     | CartPole / 离散   | Dueling 网络使用共享特征层，并组合 Value 与 Advantage：`Q = V + A - mean(A)`；采用经验回放和目标网络。            |
| `PolicyGradient` | CartPole / 离散   | REINFORCE：按完整 episode 逆序累计折扣回报，使用动作概率的对数更新策略。                                         |
| `ActorCritic`    | CartPole / 离散   | 共享的 episode 数据分别更新 `PolicyNet` 与 `ValueNet`；Actor 使用 TD 误差作为 detached advantage。      |
| `PPO`            | CartPole / 离散   | 使用 TD/GAE 优势、优势归一化和裁剪概率比的 surrogate objective；每个 episode 进行 10 次策略更新。                 |
| `TRPO`           | CartPole / 离散   | 使用 GAE、KL 约束、Hessian-vector product、共轭梯度和线搜索更新策略；Critic 使用 Adam 更新。                   |
| `DDPG`           | Pendulum / 一维连续 | Actor-Critic、经验回放、目标 Actor/Critic、Gaussian 探索噪声衰减以及软更新。                               |
| `SAC`            | Pendulum / 一维连续 | 双 Q Critic、目标 Critic、重参数化 tanh-Gaussian 策略、可训练温度 `alpha` 与软更新。                        |

离散策略网络 `PolicyNet` 输出 softmax 概率，`Categorical` 提供采样、众数、对数概率、熵及 KL 散度计算。连续策略的 `PolicyNetCont` 和 `SACPolicyNetCont` 会依据环境动作上界缩放输出。

## 经验回放与网络工具

- `DeepQNetwork.cpp` 中维护了全局回放列表，最大容量为 10,000；DQN、Dueling DQN、DDPG 和 SAC 共用该回放数据。
- `ReplayBuffer::sample(...)` 随机抽取不重复样本；DQN/Dueling DQN 的批量大小为 80，DDPG/SAC 为 64。
- `CopyModuleParameters(...)` 用于初始同步或硬同步网络参数。
- `ParametersToVector(...)` 与 `VectorToParameters(...)` 用于 TRPO 的参数向量计算和回写。

## 目录与文件职责

| 文件                      | 职责                                      |
| ----------------------- | --------------------------------------- |
| `AdvancedChapterRL.cpp` | 程序入口和算法选择。                              |
| `BaseAdvanced.h/.cpp`   | 环境持有、通用训练/评估循环、`Categorical` 分布与模块参数工具。 |
| `CartPoleEnv.h/.cpp`    | CartPole 动力学、随机数工具、样本与 Tensor 转换工具。     |
| `PendulumEnv.h/.cpp`    | Pendulum-v1 风格的连续控制环境。                  |
| `DeepQNetwork.h/.cpp`   | DQN、Double DQN、全局经验回放及 Q 网络。            |
| `DuelingDQN.h/.cpp`     | Dueling DQN 网络与训练流程。                    |
| `PolicyGradient.h/.cpp` | REINFORCE 策略梯度实现。                       |
| `ActorCritic.h/.cpp`    | 单步 TD Actor-Critic 实现。                  |
| `PPO.h/.cpp`            | PPO 与 GAE 实现。                           |
| `TRPO.h/.cpp`           | TRPO、共轭梯度、HVP 与线搜索实现。                   |
| `DDPG.h/.cpp`           | DDPG 的连续 Actor/Critic 网络和训练流程。          |
| `SAC.h/.cpp`            | SAC 的高斯策略、双 Critic 和温度优化实现。             |
| `pch.h/.cpp`            | 预编译头及项目模块聚合引用。                          |
| `CMakeLists.txt`        | libtorch 查找、可执行目标和 C++20 构建配置。          |

## 使用限制与注意事项

- 当前训练逻辑面向示例与学习用途，并未提供模型保存、命令行参数、指标持久化或单元测试。
- `main()` 中一次只应启用一个算法的训练调用，避免多个模型依次运行导致耗时和输出混杂。
- `DDPG` 与 `SAC` 均通过 `TORCH_CHECK` 限制为一维连续动作；扩展到多维动作前需要同步调整动作采样、环境接口和返回值。
- 回放缓冲区为全局数据。每个使用回放的算法会在训练开始时清空它，但不支持并发训练多个此类算法。
- TRPO 与 PPO 的 GAE 实现会将 TD 张量移至 CPU 后逐项计算，再移回原设备；这会在 GPU 场景产生额外的数据传输开销。
- CartPole 的 episode 上限由 `BaseAdvanced` 中的 `m_maxMewardCount` 控制；连续控制算法训练时将其设置为 200，以匹配 Pendulum 的时间限制。
  
  

# 作者

**qq：** 2907078601 
**name：** colin 




