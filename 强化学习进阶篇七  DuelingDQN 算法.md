# DuelingDQN

## DuelingDQN 由来

标准 DQN 使用一个网络直接输出每个动作的动作价值：$Q(s,a)$。但是在一些状态下，环境当前的好坏比具体选择哪个动作更重要。例如车杆已经接近倾倒时，无论向左还是向右施力，状态本身的价值都比较低；而在车杆平衡时，两个动作的价值可能很接近。

DuelingDQN（Dueling Deep Q-Network）将 $Q$ 值拆分为两部分：

- **状态价值函数 $V(s)$**：只评价当前状态 $s$ 的好坏；
- **优势函数 $A(s,a)$**：表示在状态 $s$ 下，动作 $a$ 相比其他动作带来的额外价值。

这样网络不需要在每个状态都分别学习所有动作的完整 $Q$ 值，而是先学习状态价值，再学习动作之间的差异。对于动作价值相近的状态，这种结构通常更有效。



## DuelingDQN 公式推导

动作价值函数可以写为：

$Q(s,a)=V(s)+A(s,a)$

但这个式子存在不可辨识问题：对于任意常数 $c$，将 $V(s)$ 增加 $c$，同时将所有 $A(s,a)$ 减少 $c$，得到的 $Q(s,a)$ 不变。因此不能直接使用这个式子组合两个分支。

DuelingDQN 对优势函数减去所有动作优势的平均值：

$Q(s,a)=V(s)+\left(A(s,a)-\frac{1}{|\mathcal A|}\sum_{a'}A(s,a')\right)$

其中：

- $|\mathcal A|$ 是动作空间大小；
- $V(s)$ 的输出维度为 $1$；
- $A(s,a)$ 的输出维度为动作数量；
- 每个样本的优势均值为 $\frac{1}{|\mathcal A|}\sum_{a'} A(s,a')$。

经过聚合后，网络最终输出仍然是每个动作对应的 $Q$ 值：

$Q(s,\cdot)=[Q(s,a_0),Q(s,a_1),\ldots]$

DuelingDQN 改变的是网络结构，DQN 的 TD 目标和损失函数不变：

$y=r+\gamma\max_{a'}Q_{\theta'}(s',a')(1-done)$

$\mathcal L_{\text{MSE}}=\left(y-Q_{\theta}(s,a)\right)^2$

其中在线网络 $Q_{\theta}$ 预测当前动作价值，目标网络 $Q_{\theta'}$ 计算 TD 目标。

**总结：** DuelingDQN 用一个共享特征层提取状态特征，再分别估计 $V(s)$ 和 $A(s,a)$，最后聚合为 $Q(s,a)$。经验回放、$\epsilon$-贪心、目标网络和 DQN 的训练方式保持一致。



## DuelingDQN 实现细节

### 0. Dueling 网络

网络先通过全连接层提取特征，然后分为两个分支：

- `m_V`：价值分支，输出每个状态的 $V(s)$；
- `m_A`：优势分支，输出所有动作的 $A(s,a)$。

```
class DuelingNetImpl : public torch::nn::Module
{
public:
    DuelingNetImpl() = default;
    DuelingNetImpl(int64_t input, int64_t output, int64_t hidden = 128)
    {
        m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
        m_A = register_module("A", torch::nn::Linear(hidden, output));
        m_V = register_module("V", torch::nn::Linear(hidden, 1));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
        auto a = m_A->forward(x);
        auto v = m_V->forward(x);
        return v + a - a.mean(1).view({ -1, 1 });
    }

    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_V{ nullptr };
    torch::nn::Linear m_A{ nullptr };
};

TORCH_MODULE(DuelingNet);
```

`a.mean(1)` 计算每个样本所有动作优势值的平均值。`view({ -1, 1 })` 将它转换为列向量，使其能与所有动作的优势值相减。

例如某状态下：

$V(s)=3,\quad A(s,\cdot)=[1,3]$

优势均值为 $2$，最终输出为：

$Q(s,\cdot)=3+[1,3]-2=[2,4]$



### 1. 创建在线网络、目标网络和优化器

```
auto input = m_objEnv->GetStateDim();
auto output = m_objEnv->GetActionDim();

m_Qnet = DuelingNet(input, output);
m_TargetQnet = DuelingNet(input, output);
m_Qnet->to(m_device);
m_TargetQnet->to(m_device);

CreateOptimizer(m_Qnet);
SyncTargetNet();
```

1. `m_Qnet` 是在线网络，用于选择动作并参与反向传播；
2. `m_TargetQnet` 是目标网络，用于计算稳定的 TD 目标；
3. `SyncTargetNet()` 将在线网络参数复制到目标网络，保证训练开始时两个网络一致。
   
   

### 2. 实现动作：$\epsilon$-贪心策略

```
double DuelingDQN::TakeAction(VectorDouble& s0, bool bPredict)
{
    int a = 0;
    if (!bPredict && m_xRandomData.RandDouble(0, 1.0) < m_dbEpsilon)
    {
        a = m_xRandomData.RandInt(0, 1);
    }
    else
    {
        torch::NoGradGuard no_grad;
        auto s = VectorDoubleTensor(s0, m_device);
        auto q = m_Qnet->forward(s);
        a = q.squeeze().argmax().item<int>();
    }

    return a;
}
```

训练时，按概率 $\epsilon$ 随机选择动作，增加探索；其余情况下选择网络输出中最大 $Q$ 值对应的动作。评测时 `bPredict` 为 `true`，始终选择最大 $Q$ 值动作。

虽然网络内部使用价值分支和优势分支，但 `forward()` 已经聚合得到完整的 $Q(s,a)$，因此动作选择方式和 DQN 相同。



### 3. 经验回放与网络更新

```
void DuelingDQN::Update()
{
    ReplayBuffer dataTrain;
    static int count = 1;

    auto samples = dataTrain.sample(m_batchsize);
    auto [s0, a, r, s1, done] = QwListToTensor(samples, m_device);

    auto q = m_Qnet->forward(s0).gather(1, a);

    auto [q1, _] = m_TargetQnet->forward(s1).max(1);
    q1 = q1.view({ -1, 1 });

    auto qtargets = r + m_dbGamma * q1 * (1 - done);

    auto mseloss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
    auto dqnloss = mseloss->forward(q, qtargets);
    m_pAdam->zero_grad();
    dqnloss.backward();
    m_pAdam->step();

    if (count % 10 == 0)
    {
        SyncTargetNet();
    }
    count++;
}
```

从回放缓冲区随机采样一个批次后：

1. `m_Qnet->forward(s0).gather(1, a)` 获取在线网络对实际执行动作的预测值 $Q_\theta(s,a)$；
2. `m_TargetQnet->forward(s1).max(1)` 获取下一状态所有动作中的最大目标 $Q$ 值；
3. `done` 为终止标记，终止状态不再加入下一状态价值；
4. 使用 MSE 损失更新在线网络；
5. 每更新 10 次在线网络，用 `SyncTargetNet()` 同步一次目标网络。

这里的更新目标仍是标准 DQN 目标，DuelingDQN 的区别仅在于 `m_Qnet` 和 `m_TargetQnet` 使用了双流网络结构。



### 4. 训练时机与终止条件

```
void DuelingDQN::TrainGenerateItem1(const QwItem& item)
{
    AddReplayDataList(item);
}

void DuelingDQN::TrainGenerateItem2(const QwList& vList)
{
    if (450 < vList.size())
    {
        m_bEndGenerateTrain = true;
        return;
    }

    if (m_nMinimalsize < GetReplayDataList().size())
    {
        int max = 10;
        if (100 < vList.size())
        {
            max = vList.size() / 2;
        }

        for (int i = 0; i < max; i++)
        {
            Update();
        }
    }
}
```

每一步环境交互产生的样本先加入回放缓冲区。当缓冲区样本数超过 `m_nMinimalsize`（500）后，才开始随机采样训练。

本实现不会每采集一个样本就立刻更新网络，而是在每个回合结束后集中调用多次 `Update()`：默认训练 10 次；回合奖励超过 100 时，训练次数增加到当前回合奖励的一半。

**训练终止条件：** 达到最大迭代次数，或单回合奖励超过 450。



### 5. DuelingDQN 与 DQN 的区别

| 项目             | DQN                | DuelingDQN                         |
| -------------- | ------------------ | ---------------------------------- |
| 网络输出           | 直接输出每个动作的 $Q(s,a)$ | 分别输出 $V(s)$、$A(s,a)$，再聚合为 $Q(s,a)$ |
| 状态价值估计         | 隐含在每个动作 $Q$ 值中     | 由价值分支显式估计                          |
| 动作差异估计         | 隐含在每个动作 $Q$ 值中     | 由优势分支显式估计                          |
| TD 目标与损失       | 标准 DQN             | 与标准 DQN 相同                         |
| 经验回放、目标网络、探索策略 | 使用                 | 使用                                 |

DuelingDQN 可以与 DoubleDQN、优先经验回放等改进组合使用。例如使用 DoubleDQN 的动作选择和目标估计方式，再使用 Dueling 网络输出 $Q$ 值，即可构成 Dueling Double DQN。
