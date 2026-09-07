# Actor-Critic 算法

## 算法回顾

- 车杆事件：随机初始状态 → 选择动作→下一个状态→ $\dots$ → 结束

- DQN 以及 DQN的改进算法**基于价值**的方法解决连续状态的问题，用神经网络代替$Q$表(**拟合动作价值**)。 基于最大值选择动作，算法目标是优化价值神经网络。

- 策略梯度算法是**基于策略**的方法，用神经网络对动作进行建模（分布概率），算法目标是最大化轨迹期望总回报。

![ActorCritic](ActorCritic.png)

## ActorCritic 算法

策略梯度算法使用策略网络 $\pi_\theta(a|s)$ 直接输出动作概率，并使用一个完整回合的累计折扣回报 $G_t$ 更新网络：

$\mathcal L_{Actor}=-G_t\log\pi_\theta(a_t|s_t)$

这种算法简单直观，但必须等一个回合结束后才能计算累计回报，而且蒙特卡洛回报 $G_t$ 的方差较大，训练过程容易波动。Actor-Critic 是囊括一系列算法的整体架构，目前很多高效的前沿算法都属于 Actor-Critic 算法，它既学习价值函数，又学习策略函数。

ActorCritic（演员-评论家）算法同时训练两个神经网络：

- **Actor（演员）**：策略网络 $\pi_\theta(a|s)$，根据当前状态选择动作 （**用上一章策略梯度网络**）；
- **Critic（评论家）**：价值网络 $V_\omega(s)$，评价当前状态的价值，并指导 Actor 更新 (**ValueNet与DQN中的价值网络一样**)。

可以这样理解：Actor 负责做动作，Critic 负责评价动作产生的结果。Actor 根据 Critic 给出的评价调整动作概率，Critic 根据实际奖励不断提高自己的评价准确性。

ActorCritic 不需要等待回合结束再计算完整的累计回报，而是利用当前奖励和下一状态价值构造 TD 目标，因此可以更及时地更新网络。



![ActorCritic2.png](ActorCritic2.png)





**ActorCritic 实现** 如上图 策略梯度算法更新时：

公式：$\mathcal L_{Actor}=-G_t\log\pi_\theta(a_t|s_t)$  中 由$G_t$ 换成 $TD$ ,$TD$是Critic网络中的**时序差分的误差**





### 1. 创建 Actor、Critic 和优化器

```cpp
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



### 2. Actor 选择动作

```cpp
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



### 3. 将一个回合的数据转换成张量

```
auto [s0, a, r, s1, done] = QwListToTensor(vList, m_device);
```

一个回合中每一步的数据格式为：

$(s_t,a_t,r_t,s_{t+1},done_t)$

转换完成后：

- `s0`：当前状态批次；
- `a`：实际执行的动作批次；
- `r`：即时奖励批次；
- `s1`：下一状态批次；
- `done`：回合终止标记批次。

本实现一次使用当前回合中的全部数据计算 Actor 和 Critic 的平均损失。



### 4. 更新策略

```cpp
void ActorCritic::TrainGenerateItem2(const QwList& vList)
{

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

    auto [s0, a, r, s1, done] = QwListToTensor(vList, m_device);

    auto v0 = m_CriticNet->forward(s0);
    auto v1 = r + m_dbGamma * m_CriticNet->forward(s1) * (1 - done);

    auto td = v1 - v0;

    auto action = m_ActorNet->forward(s0).gather(1, a);
    auto logProbs = torch::log(action);
    auto actorLoss = torch::mean(-logProbs * td.detach());

    auto criticLoss = torch::mean(torch::mse_loss(v0, v1.detach()));

    m_pAdamActor->zero_grad();
    m_pAdamCritic->zero_grad();

    actorLoss.backward();
    criticLoss.backward();


    m_pAdamActor->step();
    m_pAdamCritic->step();
}
```





### 5. 训练终止条件

```cpp
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



### 运行效果

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
    actorCritic.Play(1000);

}

/**

Currently Actor-Critic
GenerateTrainData .....
train i: 80 / 1000 , rewardCount: 11
train i: 100 / 1000 , rewardCount: 15
train i: 120 / 1000 , rewardCount: 41
train i: 140 / 1000 , rewardCount: 13
train i: 160 / 1000 , rewardCount: 50
train i: 180 / 1000 , rewardCount: 37
train i: 200 / 1000 , rewardCount: 19
train i: 220 / 1000 , rewardCount: 29
train i: 240 / 1000 , rewardCount: 51
train i: 260 / 1000 , rewardCount: 76
train i: 280 / 1000 , rewardCount: 38
train i: 300 / 1000 , rewardCount: 49
train i: 320 / 1000 , rewardCount: 124
train i: 340 / 1000 , rewardCount: 54
train i: 360 / 1000 , rewardCount: 148
train i: 380 / 1000 , rewardCount: 95
train i: 400 / 1000 , rewardCount: 180
train i: 420 / 1000 , rewardCount: 470
train i: 440 / 1000 , rewardCount: 212
train i: 460 / 1000 , rewardCount: 233
train i: 480 / 1000 , rewardCount: 249
train i: 500 / 1000 , rewardCount: 470
train i: 520 / 1000 , rewardCount: 146
train i: 540 / 1000 , rewardCount: 394
train i: 560 / 1000 , rewardCount: 470
train i: 579, break Generate Train ####

TestData .....
count: 1 , rewardCount: 492
count: 2 , rewardCount: 500
count: 3 , rewardCount: 500
count: 4 , rewardCount: 500
count: 5 , rewardCount: 500
count: 6 , rewardCount: 500
count: 7 , rewardCount: 500
count: 8 , rewardCount: 500
count: 9 , rewardCount: 500
count: 10 , rewardCount: 500
count: 11 , rewardCount: 500
count: 12 , rewardCount: 500
count: 13 , rewardCount: 500
count: 14 , rewardCount: 500
count: 15 , rewardCount: 500
count: 16 , rewardCount: 500
count: 17 , rewardCount: 500
count: 18 , rewardCount: 500
count: 19 , rewardCount: 500
count: 20 , rewardCount: 500
count: 21 , rewardCount: 500
count: 22 , rewardCount: 500
count: 23 , rewardCount: 500
count: 24 , rewardCount: 444
count: 25 , rewardCount: 500
count: 26 , rewardCount: 492
count: 27 , rewardCount: 500
count: 28 , rewardCount: 500
count: 29 , rewardCount: 500
count: 30 , rewardCount: 410
count: 31 , rewardCount: 500
count: 32 , rewardCount: 500
count: 33 , rewardCount: 500
count: 34 , rewardCount: 500
count: 35 , rewardCount: 396
count: 36 , rewardCount: 500
count: 37 , rewardCount: 500
count: 38 , rewardCount: 500
count: 39 , rewardCount: 500
count: 40 , rewardCount: 500
count: 41 , rewardCount: 500
count: 42 , rewardCount: 421
count: 43 , rewardCount: 500
count: 44 , rewardCount: 500
count: 45 , rewardCount: 500
count: 46 , rewardCount: 500
count: 47 , rewardCount: 435
count: 48 , rewardCount: 486
count: 49 , rewardCount: 500
count: 50 , rewardCount: 500
count: 51 , rewardCount: 500
count: 52 , rewardCount: 500
count: 53 , rewardCount: 500
count: 54 , rewardCount: 500
count: 55 , rewardCount: 462
count: 56 , rewardCount: 500
count: 57 , rewardCount: 500
count: 58 , rewardCount: 500
count: 59 , rewardCount: 500
count: 60 , rewardCount: 500
count: 61 , rewardCount: 500
count: 62 , rewardCount: 391
count: 63 , rewardCount: 500
count: 64 , rewardCount: 500
count: 65 , rewardCount: 430
count: 66 , rewardCount: 500



**/

```





## ActorCritic 与策略梯度的区别
