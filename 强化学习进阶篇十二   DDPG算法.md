# DDPG 算法

## DDPG 算法由来

DQN 适用于离散动作空间。以车杆环境为例，动作只有向左和向右，可以让神经网络输出所有动作的 $Q$ 值，再选择其中最大的动作。

但是在连续动作环境中，动作可能是一个任意实数。例如控制摆锤时，力矩可以取 $[-2,2]$ 之间的任意值。此时无法枚举所有动作并计算：

$\max_a Q(s,a)$

DDPG（Deep Deterministic Policy Gradient，深度确定性策略梯度）使用 ActorCritic 结构解决连续动作问题：

- **Actor：** 根据状态直接输出一个确定性连续动作 $a=\mu_\theta(s)$；
- **Critic：** 输入状态和动作，输出动作价值 $Q_\omega(s,a)$；
- **目标 Actor：** 为下一状态生成目标动作；
- **目标 Critic：** 计算稳定的 TD 目标；
- **经验回放：** 保存历史交互数据并随机采样训练；
- **探索噪声：** 在 Actor 输出的确定性动作上加入随机噪声。

DDPG 可以理解为 DQN 与确定性策略梯度的结合。Critic 的训练方式与 DQN 类似，Actor 则沿着 Critic 认为价值更高的动作方向更新。



## DDPG 公式推导

### 1. 确定性策略

Actor 使用确定性策略：

$a=\mu_\theta(s)$

与随机策略 $\pi_\theta(a|s)$ 不同，同一个状态输入确定性 Actor 时，总是得到相同动作。

Actor 的目标是让 Critic 对其输出动作给出尽可能高的评价：

$J(\theta)=\mathbb E_{s\sim D}\left[Q_\omega(s,\mu_\theta(s))\right]$

神经网络优化器使用梯度下降，因此定义 Actor 损失：

$\mathcal L_{Actor}=-\mathbb E_{s\sim D}\left[Q_\omega(s,\mu_\theta(s))\right]$

最小化负 $Q$ 值等价于最大化 Actor 生成动作的 $Q$ 值。



### 2. Critic 的 TD 目标

DDPG 使用目标 Actor 和目标 Critic 计算 TD 目标：

$a'=\mu_{\theta'}(s')$

$y=r+\gamma Q_{\omega'}(s',a')(1-done)$

Critic 使用均方误差训练：

$\mathcal L_{Critic}=\mathbb E\left[\left(Q_\omega(s,a)-y\right)^2\right]$

其中：

- $\mu_\theta$：在线 Actor；
- $Q_\omega$：在线 Critic；
- $\mu_{\theta'}$：目标 Actor；
- $Q_{\omega'}$：目标 Critic。
  
  

### 3. 目标网络软更新

DQN 每隔若干次训练将在线网络参数完整复制到目标网络。DDPG 通常每次训练后都进行软更新：

$\theta'\leftarrow(1-\tau)\theta'+\tau\theta$

$\omega'\leftarrow(1-\tau)\omega'+\tau\omega$

其中 $\tau$ 是一个很小的数。本实现使用：

$\tau=0.005$

目标网络每次只向在线网络移动一小步，使 TD 目标变化更加平滑。



### 4. 连续动作探索

确定性策略本身不会随机探索，因此训练时在 Actor 动作上加入高斯噪声：

$a=\operatorname{clip}(\mu_\theta(s)+\epsilon,a_{low},a_{high})$

$\epsilon\sim\mathcal N(0,\sigma^2)$

评测时不加入噪声，直接使用 Actor 输出的动作。

**总结：** DDPG 使用 Actor 直接生成连续动作，使用 Critic 评价状态动作对，通过经验回放和目标网络训练，并使用噪声解决确定性策略的探索问题。



## DDPG 实现细节

### 0. 连续动作 Actor 网络

```
class PolicyNetContImpl : public torch::nn::Module
{
public:
    PolicyNetContImpl() = default;

    PolicyNetContImpl(
        int64_t input,
        int64_t output,
        double actionBound,
        int64_t hidden = 128)
    {
        m_fc1 = register_module(
            "fc1", torch::nn::Linear(input, hidden));
        m_fc2 = register_module(
            "fc2", torch::nn::Linear(hidden, output));
        m_dbActionBound = actionBound;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
        return m_dbActionBound * torch::tanh(m_fc2->forward(x));
    }

private:
    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_fc2{ nullptr };
    double m_dbActionBound = 2.0;
};

TORCH_MODULE(PolicyNetCont);
```

Actor 输入状态 $s$，输出连续动作。输出层使用 `tanh`，将值限制在 $[-1,1]$，再乘以动作上界 `actionBound`：

$a=a_{bound}\tanh(x)$

当环境动作范围为 $[-2,2]$ 时，Actor 的输出也位于该范围内。



### 1. 连续动作 Critic 网络

```
class QValueNetContImpl : public torch::nn::Module
{
public:
    QValueNetContImpl() = default;

    QValueNetContImpl(
        int64_t input,
        int64_t output,
        int64_t hidden = 128)
    {
        m_fc1 = register_module(
            "fc1", torch::nn::Linear(input + output, hidden));
        m_fc2 = register_module(
            "fc2", torch::nn::Linear(hidden, hidden));
        m_output = register_module(
            "output", torch::nn::Linear(hidden, 1));
    }

    torch::Tensor forward(
        const torch::Tensor& state,
        const torch::Tensor& action)
    {
        auto x = torch::cat({ state, action }, 1);
        x = torch::relu(m_fc1->forward(x));
        x = torch::relu(m_fc2->forward(x));
        return m_output->forward(x);
    }

private:
    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_fc2{ nullptr };
    torch::nn::Linear m_output{ nullptr };
};

TORCH_MODULE(QValueNetCont);
```

Critic 同时输入状态和动作：

$Q_\omega(s,a)$

代码使用 `torch::cat({ state, action }, 1)` 将状态与动作在特征维度拼接，然后输出一个标量 $Q$ 值。

DQN 的网络输入状态并一次输出所有离散动作的 $Q$ 值；DDPG 的 Critic 输入一个具体连续动作并输出这个动作的 $Q$ 值。



### 2. 创建四个网络和两个优化器

```
m_actor = PolicyNetCont(m_stateDim, m_actionDim, actionBound);
m_targetActor = PolicyNetCont(
    m_stateDim, m_actionDim, actionBound);

m_critic = QValueNetCont(m_stateDim, m_actionDim);
m_targetCritic = QValueNetCont(m_stateDim, m_actionDim);

m_actor->to(m_device);
m_targetActor->to(m_device);
m_critic->to(m_device);
m_targetCritic->to(m_device);

CopyModuleParameters(*m_actor, *m_targetActor);
CopyModuleParameters(*m_critic, *m_targetCritic);

m_actorOptimizer = std::make_unique<torch::optim::Adam>(
    m_actor->parameters(),
    torch::optim::AdamOptions(m_actorLearningRate));
m_criticOptimizer = std::make_unique<torch::optim::Adam>(
    m_critic->parameters(),
    torch::optim::AdamOptions(m_criticLearningRate));
```

训练开始时创建：

1. 在线 Actor `m_actor`；
2. 目标 Actor `m_targetActor`；
3. 在线 Critic `m_critic`；
4. 目标 Critic `m_targetCritic`。

目标网络的初始参数从在线网络完整复制。Actor 和 Critic 分别使用 Adam 优化器：

- Actor 学习率为 $10^{-4}$；
- Critic 学习率为 $10^{-3}$。

当前实现要求连续动作维度为 $1$：

```
TORCH_CHECK(
    m_actionDim == 1,
    "DDPG currently supports only one-dimensional continuous actions");
```



### 3. 训练和评测动作

```
double DDPG::TakeAction(VectorDouble& s0, bool bPredict)
{
    torch::NoGradGuard noGrad;

    auto s = VectorDoubleTensor(s0, m_device);
    auto actionTensor = m_actor->forward(s);
    auto action = actionTensor.squeeze().item<double>();

    if (!bPredict)
    {
        auto noise = torch::randn(
            { m_objEnv->GetActionDim() }, m_device) * m_dbSigma;
        action = action + noise.squeeze().item<double>();
        action = std::clamp(
            action,
            m_objEnv->GetActionLow(),
            m_objEnv->GetActionHigh());
    }

    return action;
}
```

训练时动作公式为：

$a=\operatorname{clip}(\mu_\theta(s)+\epsilon,a_{low},a_{high})$

评测时 `bPredict=true`，不添加探索噪声，直接使用确定性 Actor 输出。

`torch::NoGradGuard` 表示与环境交互时不构建计算图。网络训练时会从回放缓冲区重新读取状态并执行前向计算。



### 4. 经验回放与训练时机

```
void DDPG::TrainGenerateItem1(const QwItem& item)
{
    AddReplayDataList(item);

    if (m_nMinimalsize < GetReplayDataList().size())
    {
        Update();
    }
}
```

每次环境交互产生一条数据：

$(s,a,r,s',done)$

样本先加入经验回放缓冲区。当缓冲区样本数超过 `m_nMinimalsize`（1000）后，每加入一个新样本就调用一次 `Update()`。

每次更新随机采样 64 条数据：

```
ReplayBuffer dataTrain;
auto samples = dataTrain.sample(m_batchSize);
auto [s0, a, r, s1, done] =
    QwListToTensor(samples, m_device, true);
```

最后一个参数 `true` 表示处理连续动作数据。



### 5. 探索噪声衰减

```
m_dbSigma = std::max(0.02, m_dbSigma * 0.9995);
```

初始噪声标准差为：

$\sigma=0.2$

每次网络更新后按比例衰减：

$\sigma\leftarrow\max(0.02,0.9995\sigma)$

训练初期使用较大噪声进行探索，训练后期降低噪声以提高动作稳定性，同时保留最小值 $0.02$，避免完全停止探索。



### 6. 计算 Critic 的 TD 目标

```
torch::Tensor qTargets;
{
    torch::NoGradGuard noGrad;

    auto q1 = m_targetCritic->forward(
        s1, m_targetActor->forward(s1));

    qTargets = r + m_dbGamma * q1 * (1.0 - done);
}
```

目标 Actor 为下一状态生成动作：

$a'=\mu_{\theta'}(s')$

目标 Critic 评价该动作：

$Q_{\omega'}(s',a')$

最终 TD 目标为：

$y=r+\gamma Q_{\omega'}(s',\mu_{\theta'}(s'))(1-done)$

`torch::NoGradGuard` 保证目标网络不参与反向传播。



### 7. 更新 Critic

```
auto mseloss = torch::nn::MSELoss(
    torch::nn::MSELossOptions().reduction(torch::kMean));
auto criticLoss = mseloss->forward(
    m_critic->forward(s0, a), qTargets);

m_criticOptimizer->zero_grad();
criticLoss.backward();
m_criticOptimizer->step();
```

Critic 损失为：

$\mathcal L_{Critic}=\operatorname{MSE}(Q_\omega(s,a),y)$

其中动作 $a$ 是回放缓冲区中实际执行的动作。Critic 通过监督学习逐渐逼近 TD 目标。



### 8. 更新 Actor

```
auto actorLoss =
    -m_critic->forward(s0, m_actor->forward(s0)).mean();

m_criticOptimizer->zero_grad();
m_actorOptimizer->zero_grad();
actorLoss.backward();
m_actorOptimizer->step();
```

Actor 首先根据状态生成动作：

$a=\mu_\theta(s)$

再由在线 Critic 评价动作：

$Q_\omega(s,\mu_\theta(s))$

Actor 损失为：

$\mathcal L_{Actor}=-\mathbb E\left[Q_\omega(s,\mu_\theta(s))\right]$

反向传播时，梯度从 Critic 输出经过动作输入继续传播到 Actor，使 Actor 学习生成更高 $Q$ 值的动作。

代码只调用 `m_actorOptimizer->step()`，因此本阶段只更新 Actor 参数。清空 Critic 梯度可以避免 Actor 反向传播产生的 Critic 梯度残留到下一次训练。



### 9. 软更新目标网络

```
void DDPG::SoftUpdate(
    torch::nn::Module& source,
    torch::nn::Module& target)
{
    torch::NoGradGuard noGrad;

    auto sourceParameters = source.parameters();
    auto targetParameters = target.parameters();

    for (size_t i = 0; i < sourceParameters.size(); ++i)
    {
        targetParameters[i].mul_(1.0 - m_tau);
        targetParameters[i].add_(sourceParameters[i], m_tau);
    }
}
```

每次更新 Actor 和 Critic 后执行：

```
SoftUpdate(*m_actor, *m_targetActor);
SoftUpdate(*m_critic, *m_targetCritic);
```

对应公式：

$\theta'\leftarrow(1-\tau)\theta'+\tau\theta$

$\omega'\leftarrow(1-\tau)\omega'+\tau\omega$

本实现中 $\tau=0.005$，即目标网络保留 $99.5\%$ 的旧参数，只融合 $0.5\%$ 的在线网络新参数。



### 10. 完整训练流程

DDPG 的一次环境交互与训练过程为：

1. 在线 Actor 根据状态输出连续动作；
2. 训练时向动作加入高斯探索噪声并裁剪到合法范围；
3. 执行动作，将 $(s,a,r,s',done)$ 加入经验回放；
4. 回放缓冲区超过 1000 条数据后随机采样 64 条；
5. 目标 Actor 和目标 Critic 计算 TD 目标；
6. 使用 MSE 损失更新在线 Critic；
7. 最大化在线 Critic 的 $Q$ 值来更新在线 Actor；
8. 对 Actor 和 Critic 的目标网络进行软更新；
9. 逐渐减小探索噪声并继续采样。
   
   

## DDPG 与 DQN 的区别

| 项目        | DQN           | DDPG                 |
| --------- | ------------- | -------------------- |
| 动作空间      | 离散动作          | 连续动作                 |
| 策略        | 选择最大 $Q$ 值动作  | Actor 直接输出动作         |
| Critic 输入 | 状态            | 状态和动作                |
| 在线网络      | 一个 Q 网络       | Actor + Critic       |
| 目标网络      | 目标 Q 网络       | 目标 Actor + 目标 Critic |
| 探索方式      | $\epsilon$-贪心 | 动作加入随机噪声             |
| 目标网络更新    | 定期硬同步         | 每次软更新                |
| 经验回放      | 使用            | 使用                   |

DDPG 能够处理连续动作，但确定性策略对 Critic 估计误差和超参数较敏感。后续 TD3 使用双 Critic、延迟策略更新和目标策略平滑改善 DDPG 的过估计问题；SAC 则使用随机策略和最大熵目标进一步提高探索能力与训练稳定性。
