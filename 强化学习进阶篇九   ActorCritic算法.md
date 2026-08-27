# ActorCritic 算法

## ActorCritic 算法由来

策略梯度算法使用策略网络 $\pi_\theta(a|s)$ 直接输出动作概率，并使用一个完整回合的累计折扣回报 $G_t$ 更新网络：

$$\mathcal L_{Actor}=-G_t\log\pi_\theta(a_t|s_t)$$

这种 REINFORCE 算法简单直观，但必须等一个回合结束后才能计算累计回报，而且蒙特卡洛回报 $G_t$ 的方差较大，训练过程容易波动。

ActorCritic（演员-评论家）算法同时训练两个神经网络：

- **Actor（演员）**：策略网络 $\pi_\theta(a|s)$，根据当前状态选择动作；
- **Critic（评论家）**：价值网络 $V_\omega(s)$，评价当前状态的价值，并指导 Actor 更新。

可以这样理解：Actor 负责做动作，Critic 负责评价动作产生的结果。Actor 根据 Critic 给出的评价调整动作概率，Critic 根据实际奖励不断提高自己的评价准确性。

ActorCritic 不需要等待回合结束再计算完整的累计回报，而是利用当前奖励和下一状态价值构造 TD 目标，因此可以更及时地更新网络。



## ActorCritic 公式推导

### 1. Critic 的 TD 目标

Critic 使用神经网络估计状态价值：

$$V_\omega(s)=\mathbb E_{\pi}\left[G_t|s_t=s\right]$$

根据贝尔曼方程，当前状态价值的单步 TD 目标为：

$$y_t=r_t+\gamma V_\omega(s_{t+1})(1-done_t)$$

其中：

- $r_t$ 是执行动作后获得的即时奖励；
- $\gamma$ 是折扣因子；
- $V_\omega(s_{t+1})$ 是 Critic 对下一状态价值的估计；
- $done_t=1$ 表示回合已经终止，终止状态没有后续价值。

TD Error（时序差分误差）为：

$$\delta_t=y_t-V_\omega(s_t)$$

即：

$$\delta_t=r_t+\gamma V_\omega(s_{t+1})(1-done_t)-V_\omega(s_t)$$

Critic 使用均方误差训练：

$$\mathcal L_{Critic}=\frac{1}{N}\sum_t\left(y_t-V_\omega(s_t)\right)^2$$



### 2. Actor 的策略梯度

策略梯度使用优势函数评价动作：

$$\nabla_\theta J(\theta)=\mathbb E\left[A(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t|s_t)\right]$$

在 ActorCritic 中，可以使用 TD 误差 $\delta_t$ 作为优势函数 $A(s_t,a_t)$ 的近似：

$$A(s_t,a_t)\approx\delta_t$$

因此 Actor 的损失函数为：

$$\mathcal L_{Actor}=-\frac{1}{N}\sum_t\log\pi_\theta(a_t|s_t)\delta_t$$

公式含义：

1. 当 $\delta_t>0$ 时，实际结果比 Critic 原来的预期更好，增大动作 $a_t$ 的概率；
2. 当 $\delta_t<0$ 时，实际结果比预期更差，减小动作 $a_t$ 的概率；
3. $|\delta_t|$ 越大，Actor 更新幅度越大。

**总结：** Critic 通过最小化 TD 目标与当前价值估计之间的误差来学习 $V(s)$；Actor 使用 Critic 计算的 TD 误差作为动作优势，更新策略网络。



## ActorCritic 实现细节

### 0. Actor 策略网络

Actor 使用前面策略梯度算法中的 `PolicyNet`：

```
class PolicyNetImpl : public torch::nn::Module
{
public:
    PolicyNetImpl() = default;

    PolicyNetImpl(int64_t input, int64_t output, int64_t hidden = 128)
    {
        m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
        m_fc2 = register_module("fc2", torch::nn::Linear(hidden, output));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
        x = m_fc2->forward(x);
        return torch::softmax(x, 1);
    }

    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_fc2{ nullptr };
};

TORCH_MODULE(PolicyNet);
```

Actor 的输入是状态 $s$，输出是所有离散动作的概率：

$$\pi_\theta(\cdot|s)=[P(a_0|s),P(a_1|s),\ldots]$$

输出层使用 `softmax`，保证每个动作概率大于等于 $0$，并且所有动作概率之和为 $1$。



### 1. Critic 价值网络

```
class ValueNetImpl : public torch::nn::Module
{
public:
    ValueNetImpl() = default;

    ValueNetImpl(int64_t input, int64_t output = 1, int64_t hidden = 128)
    {
        m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
        m_fc2 = register_module("fc2", torch::nn::Linear(hidden, output));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
        return m_fc2->forward(x);
    }

    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_fc2{ nullptr };
};

TORCH_MODULE(ValueNet);
```

Critic 输入状态 $s$，输出一个标量 $V_\omega(s)$，表示从当前状态出发，按照 Actor 的策略继续执行时能够获得的预期累计折扣奖励。

Critic 输出层不使用 `softmax`，因为状态价值不是概率，可以是任意实数。



### 2. 创建 Actor、Critic 和优化器

```
void ActorCritic::GenerateTrainData(int maxCount)
{
    cout << "Currently Actor-Critic" << endl;

    m_dbGamma = 0.98;

    auto input = m_objEnv->GetStateDim();
    auto output = m_objEnv->GetActionDim();

    m_ActorNet = PolicyNet(input, output);
    m_CriticNet = ValueNet(input, 1);

    m_CriticNet->to(m_device);
    m_ActorNet->to(m_device);

    m_pAdamActor = new torch::optim::Adam(
        m_ActorNet->parameters(), { m_dbActorLR });
    m_pAdamCritic = new torch::optim::Adam(
        m_CriticNet->parameters(), { m_dbCriticLR });

    m_ActorNet->train();
    m_CriticNet->train();

    BaseAdvanced::GenerateTrainData(maxCount);

    m_ActorNet->eval();
    m_CriticNet->eval();

    delete m_pAdamActor;
    m_pAdamActor = nullptr;

    delete m_pAdamCritic;
    m_pAdamCritic = nullptr;
}
```

Actor 和 Critic 是两个独立的神经网络，并分别使用 Adam 优化器：

- Actor 学习率 `m_dbActorLR = 1e-3`；
- Critic 学习率 `m_dbCriticLR = 1e-2`；
- 折扣因子 $\gamma=0.98$。

Critic 需要尽快学习出较准确的价值估计，因此本实现为 Critic 设置了比 Actor 更大的学习率。

ActorCritic 不需要 DQN 中的目标网络，也不需要使用 $\epsilon$-贪心策略。



### 3. Actor 选择动作

```
double ActorCritic::TakeAction(VectorDouble& s0, bool bPredict)
{
    torch::NoGradGuard no_grad;
    auto s = VectorDoubleTensor(s0, m_device);
    auto logits = m_ActorNet->forward(s);
    torch::Tensor action;

    if (bPredict)
    {
        action = logits.argmax(-1);
    }
    else
    {
        Categorical categorical(logits);
        action = categorical.sample();
    }

    return action.item<int>();
}
```

使用 `bPredict` 区分训练和评测：

- **训练时：** 按照 Actor 输出的概率分布随机采样动作，保持策略探索能力；
- **评测时：** 选择概率最大的动作。

`torch::NoGradGuard` 表示环境交互阶段不记录梯度。训练时会根据保存的状态和动作重新执行网络前向计算。



### 4. 将一个回合的数据转换成张量

```
auto [s0, a, r, s1, done] = QwListToTensor(vList, m_device);
```

一个回合中每一步的数据格式为：

$$(s_t,a_t,r_t,s_{t+1},done_t)$$

转换完成后：

- `s0`：当前状态批次；
- `a`：实际执行的动作批次；
- `r`：即时奖励批次；
- `s1`：下一状态批次；
- `done`：回合终止标记批次。

本实现一次使用当前回合中的全部数据计算 Actor 和 Critic 的平均损失。



### 5. Critic 计算 TD 目标和 TD 误差

```
auto v0 = m_CriticNet->forward(s0);
auto v1 = r + m_dbGamma * m_CriticNet->forward(s1) * (1 - done);

auto td = v1 - v0;
```

代码中：

$$v0=V_\omega(s_t)$$

$$v1=r_t+\gamma V_\omega(s_{t+1})(1-done_t)$$

$$td=v1-v0$$

`done` 为 $1$ 时，`1 - done` 为 $0$，TD 目标只保留终止动作获得的即时奖励：

$$y_t=r_t$$

`done` 为 $0$ 时，TD 目标包含下一状态的估计价值：

$$y_t=r_t+\gamma V_\omega(s_{t+1})$$



### 6. Actor 损失函数

```
auto action = m_ActorNet->forward(s0).gather(1, a);
auto logProbs = torch::log(action);
auto actorLoss = torch::mean(-logProbs * td.detach());
```

`m_ActorNet->forward(s0)` 输出每个状态下所有动作的概率，`gather(1, a)` 取出轨迹中实际执行动作的概率：

$$\pi_\theta(a_t|s_t)$$

Actor 损失为：

$$\mathcal L_{Actor}=-\operatorname{mean}\left(\log\pi_\theta(a_t|s_t)\delta_t\right)$$

`td.detach()` 非常重要。TD 误差由 Critic 计算，但更新 Actor 时只把它作为评价动作好坏的固定权重，不允许 Actor 损失的梯度传播到 Critic 网络。

因此 `actorLoss.backward()` 只更新 Actor 对应的梯度。



### 7. Critic 损失函数

```
auto criticLoss = torch::mean(torch::mse_loss(v0, v1.detach()));
```

Critic 使用均方误差：

$$\mathcal L_{Critic}=\operatorname{MSE}(V_\omega(s_t),y_t)$$

`v1.detach()` 将 TD 目标视为固定标签，阻止梯度通过下一状态价值 $V_\omega(s_{t+1})$ 继续传播。这样 Critic 只调整当前状态的预测值 `v0`，使其接近 TD 目标 `v1`。



### 8. 更新 Actor 和 Critic

```
m_pAdamActor->zero_grad();
m_pAdamCritic->zero_grad();

actorLoss.backward();
criticLoss.backward();

m_pAdamActor->step();
m_pAdamCritic->step();
```

一次训练过程如下：

1. 清空 Actor 和 Critic 上一次更新留下的梯度；
2. 反向传播 Actor 损失；
3. 反向传播 Critic 损失；
4. Actor 优化器更新策略网络；
5. Critic 优化器更新价值网络。

两个网络使用独立参数和独立优化器。Actor 依赖 Critic 的 TD 误差进行学习，但通过 `detach()` 隔离了两个损失之间不需要的梯度传播。



### 9. 训练终止条件

```
static int count = 0;

if (450 < vList.size())
{
    count++;
    if (3 < count)
    {
        m_bEndGenerateTrain = true;
        return;
    }
}
else
{
    count = 0;
}
```

当单回合步数超过 450 时，说明策略已经能够让车杆保持较长时间。本实现使用 `count` 记录连续达标次数：

- 回合步数超过 450，`count` 加 $1$；
- 回合未达到 450，`count` 清零；
- 连续 4 个回合超过 450 后终止训练。

**训练终止条件：** 达到最大迭代次数，或连续 4 个回合的步数超过 450。



## ActorCritic 与策略梯度的区别

| 项目 | REINFORCE | ActorCritic |
| --- | --- | --- |
| 策略网络 | Actor | Actor |
| 价值网络 | 无 | Critic |
| 动作评价 | 蒙特卡洛累计回报 $G_t$ | TD 误差 $\delta_t$ |
| 是否等待完整回报 | 是 | 不需要完整累计回报 |
| 更新目标 | $-G_t\log\pi(a_t|s_t)$ | $-\delta_t\log\pi(a_t|s_t)$ |
| 方差 | 较大 | 通常较小 |
| 偏差 | 蒙特卡洛估计偏差较小 | TD 自举会引入一定偏差 |

ActorCritic 使用 Critic 作为策略梯度的基线，降低了梯度估计的方差；同时使用下一状态的估计价值进行自举，不必完全依赖回合结束后的实际累计回报。

其基本训练关系为：

$$Actor:\quad\max_\theta\log\pi_\theta(a_t|s_t)\delta_t$$

$$Critic:\quad\min_\omega\left(r_t+\gamma V_\omega(s_{t+1})-V_\omega(s_t)\right)^2$$

ActorCritic 是 A2C、A3C、PPO、DDPG、SAC 等现代强化学习算法的重要基础。
