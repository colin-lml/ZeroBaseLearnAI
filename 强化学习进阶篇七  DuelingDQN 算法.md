# DuelingDQN

Dueling DQN 是 DQN 另一种的改进算法，它在 DQN 的基础上对神经网络$Q_{\theta}$进行了微小的改动，其他的和DQN 完全一样， 它将DQN神经网络$Q_{\theta}$结构 拆分为两部分：

- **状态价值函数 $V(s)$**：只评价当前状态 $s$ 的好坏；
- **优势函数 $A(s,a)$**：表示在状态 $s$ 下，动作 $a$ 相比其他动作带来的额外价值。

这样网络不需要在每个状态都分别学习所有动作的完整 $Q$ 值，而是先学习状态价值，再学习动作之间的差异。对于动作价值相近的状态，这种结构通常更有效。



## DuelingDQN 公式推导

DuelingDQN 对优势函数减去所有动作优势的平均值：

$Q(s,a)=V(s)+\left(A(s,a)-\frac{1}{|\mathcal A|}\sum_{a'}A(s,a')\right)$

其中：

- $|\mathcal A|$ 是动作空间大小；
- $V(s)$ 的输出维度为 $1$；
- $A(s,a)$ 的输出维度为动作数量；
- 每个样本的优势均值为 $\frac{1}{|\mathcal A|}\sum_{a'} A(s,a')$。

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
   
   

### 2. 经验回放与网络更新

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



### 3. 训练时机与终止条件

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

### 4. 运行效果

```
int main()
{
    DeepQNetwork  deepQN;
    //deepQN.Play(400);
    //deepQN.DoubleDQN(400);
    DuelingDQN duelingDQN;
    duelingDQN.Play(400);
 }

 /***


 ***/

```



### 5. DuelingDQN 与 DQN 的区别

| 项目             | DQN                | DuelingDQN                         |
| -------------- | ------------------ | ---------------------------------- |
| 网络输出           | 直接输出每个动作的 $Q(s,a)$ | 分别输出 $V(s)$、$A(s,a)$，再聚合为 $Q(s,a)$ |
| 状态价值估计         | 隐含在每个动作 $Q$ 值中     | 由价值分支显式估计                          |
| 动作差异估计         | 隐含在每个动作 $Q$ 值中     | 由优势分支显式估计                          |
| TD 目标与损失       | 标准 DQN             | 与标准 DQN 相同                         |
| 经验回放、目标网络、探索策略 | 使用                 | 使用                                 |








