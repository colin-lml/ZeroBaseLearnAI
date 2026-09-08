# ActorCritic算法

**Actor‑Critic 是现代主流深度强化学习算法的基础骨架，是承上启下的枢纽**，TRPO、PPO、SAC、DDPG 全部是在 Actor‑Critic 这个骨架上修改目标函数、损失、正则而来。Actor‑Critic 融合了两大路线：

1. 策略梯度算法，用神经网络对动作进行建模（分布概率），算法目标是最大化轨迹期望总回报。

2. 基于价值方法，用神经网络代替$Q$表(**拟合动作价值**)。 基于最大值选择动作，算法目标是优化价值神经网络。

回顾车杆游戏整个流程：随机初始状态 → 选择动作→下一个状态→ $\dots$ → 结束。 价值方法用价值网络拟合动作价值然后选择最优的动作，策略梯度用策略网络直接给出最优的动作。

![ActorCritic](ActorCritic.png)



Actor‑Critic是融合了价值方法和策略梯度 既学习价值函数，又学习策略函数，以策略梯度为主线，价值函数为副线（辅助）Actor（演员）和Critic（评论家）。

## Actor‑Critic更新公式

策略梯度算法使用策略网络 $\pi_\theta(a|s)$ 直接输出动作概率，并使用一个完整回合的累计折扣回报 $G_t$ 更新网络：$\mathcal L_{Actor}=-G_t\log\pi_\theta(a_t|s_t)$ 。结合Critic价值网络之后，不需要等待回合结束再计算完整的累计回报。用Critic价值网络**时序差分误差TD代替$G_t$** 如图
![ActorCritic2.png](ActorCritic2.png)

### 更新公式代码

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

### 创建 Actor、Critic 和优化器

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


