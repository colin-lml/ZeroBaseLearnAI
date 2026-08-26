# DQN

## DQN 由来

 在Q-learning 算法中用一张存储每个状态下所有动作$Q$值的表格，但在车杆环境中它的状态值就是连续的（无限个状态），  动作值是离散的(向左施加力或向右施加力)，因此不能用$Q$表，使用**函数拟合**来求解，用一神经网络来表示函数$Q$

神经网络的输入是状态$S$，输出是动作$a$对应的价值 （预期累计折扣奖励），这是DQN（Deep Q-Network，深度 Q 网络）的由来。



## DQN公式推导

回顾一下 Q-learning 的更新规则： $Q(s,a) \leftarrow Q(s,a)+ \alpha[r+ \gamma \max \limits_{a} Q(s',a')-Q(s,a)]$ ,我们用神经网络代替了$Q$表，记$Q_{dqn}$表示

1. 公式： $Q_{\theta}(s,a) \leftarrow Q_{\theta}(s,a)+ \alpha[r+ \gamma \max \limits_{a} Q_{\theta}(s',a')-Q_{\theta}(s,a)]$

2. 因为$Q_{dqn}$是神经网络所以要更新只能更新神经网络权重，也就是训练神经网络。

3. 中括号部分叫作 TD Error（时序差分误差）：$\delta=\underbrace{r+ \gamma \max \limits_{a} Q_{\theta'}(s',a')}_{目标值(标签)}-\underbrace{Q_{\theta}(s,a)}_{当前预测值}$ ，使用两套 一样 网络，**在线网络$Q_{\theta}$**   预测当前值，实时更新。**目标网络$Q_{\theta'}$** 表示目标值，定期更新。可以这样理解在Q-learning 中更新迭代$Q$表时 先更新当前的$Q(a,s)$，再更新下个$Q(a',s')$ 

4. 定义损失函数 $\mathcal L_{\text{MSE}} = \big(\underbrace{r  +\gamma Q_{\theta'}(s',a')}_{目标值} - Q_{\theta}(s,a)\big)^2$

**总结：**  使用神经网络$Q_{\theta}$ 代替$Q$表，损失函数用均方误差$MSE$ $\mathcal L_{\text{MSE}} = \big(r +\gamma Q_{\theta'}(s',a') - Q_{\theta}(s,a)\big)^2$  训练神经网络。



## DQN实现细节

### 0. 神经网络$Q_{\theta}$

```
class DQNQnetImpl : public torch::nn::Module
{
public:
    DQNQnetImpl() = default;
    DQNQnetImpl(int64_t input, int64_t output, int64_t hidden=128)
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

TORCH_MODULE(DQNQnet);

```



### 1.  经验回放

采样得到的数据存放起来维护一个**回放缓冲区**，采样数据这一时刻的状态和上一时刻的状态有关，非独立同分布的数据对训练神经网络有很大的影响，从回放缓冲区中**随机采样若干数据**来进行训练，最后将数据转换成张量。

```
    ReplayBuffer dataTrain; 
    auto samples = dataTrain.sample(m_batchsize);
    auto [s0, a, r, s1, done] = QwListToTensor(samples, m_device);
```

### 2. 创建网络和优化器 重写GenerateTrainData()接口

```
    auto input = m_objEnv->GetStateDim();
    auto output = m_objEnv->GetActionDim();

    m_Qnet = DQNQnet(input, output);
    m_TargetQnet = DQNQnet(input, output);
    m_Qnet->to(m_device);
    m_TargetQnet->to(m_device);

    CreateOptimizer(m_Qnet);

    SyncTargetNet();


```

1. 创建在线网络(m_Qnet)和目标网络(m_TargetQnet )

2. 创建线网络优化器 ,在线网络定期更新目标网络 SyncTargetNet()
   
   
   
   

### 3.实现动作 重写 TakeAction()接口

```
double DeepQNetwork::TakeAction(VectorDouble& s0, bool bPredict)
{
    int a = 0;
    if (!bPredict && m_xRandom.RandDouble(0, 1.0) < m_dbEpsilon)
    {
        a = m_xRandom.RandInt(0, 1);
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

用 bPredict变量区分训练和评测

训练时用 ϵ−贪心算法

评测时选最大$Q$值

网络输入状态 $s$，输出所有动作的 Q 值,例如：$q=Q_{\theta}=[Q_{\theta}(s,a_0),Q_{\theta}(s,a_1)]=[2,4]$

**下标$0$** 向左的预期累计奖励是 2；

**下标$1$** 向右的预期累计奖励是 4;



### 4. 实现DQN算法 重写TrainGenerateItem1/2接口

```
void DeepQNetwork::Update()
{

    ReplayBuffer dataTrain; 
    static int count = 1;

    auto samples = dataTrain.sample(m_batchsize);

    auto [s0, a, r, s1, done] = QwListToTensor(samples, m_device);

    auto q = m_Qnet->forward(s0);
    q = q.gather(1, a);

    torch::Tensor q1 ;

        q1 = m_TargetQnet->forward(s1);
    auto [qv, _] = q1.max(1);
        q1 = qv.view({ -1,1 });

    auto qtargets = r + m_dbGamma * q1 * (1 - done);

    auto mseloss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
    auto dqnloss = mseloss->forward(q, qtargets);
    m_pAdam->zero_grad();
    dqnloss.backward();
    m_pAdam->step();
    auto loss  = dqnloss.item<double>();
    if (count % 10 == 0)
    {
        SyncTargetNet();
        //cout << "dqnloss: " << loss << endl;
    }
    count++;


}


```

损失函数 均方误差$MSE$ ： $\mathcal L_{\text{MSE}} = \big(r +\gamma Q_{\theta'}(s',a') - Q_{\theta}(s,a)\big)^2$ 

代码中 $Q_{\theta'}(s',a') == m\_TargetQnet，Q_{\theta}(s,a)==m\_Qnet$

count % 10 == 0 用意 **在线网络**每更新10次 同步 **目标网络**一次

**标准DQN实现是每采样一次就要训练网络，这样计算量比较大运行测试也慢，实际实现中做了减法操作** 

```

void DeepQNetwork::TrainGenerateItem1(const QwItem& item)
{
    AddReplayDataList(item);
        /// 应该在这里实现DQN
       ///   Update();

}

void DeepQNetwork::TrainGenerateItem2(const QwList& vList)
{

    if (400 < vList.size())
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



**训练终止条件： 达迭代最大次数 或 回合数400分以上**





### 5. 运行效果

```
int main()
{
    DeepQNetwork  deepQN;
    deepQN.Play(400); 
}
/****
urrently DQN                                                                                                           
GenerateTrainData .....                                                                                                 
train i: 80 / 400 , rewardCount: 8                                                                                      
train i: 100 / 400 , rewardCount: 63                                                                                    
train i: 120 / 400 , rewardCount: 60                                                                                    
train i: 140 / 400 , rewardCount: 63                                                                                    
train i: 160 / 400 , rewardCount: 152                                                                                   
train i: 180 / 400 , rewardCount: 164                                                                                   
train i: 200 / 400 , rewardCount: 149                                                                                   
train i: 220 / 400 , rewardCount: 192                                                                                   
train i: 240 / 400 , rewardCount: 225                                                                                   
train i: 260 / 400 , rewardCount: 46                                                                                    
train i: 280 / 400 , rewardCount: 154                                                                                   
train i: 297, break Generate Train ####                                                                                 

TestData .....                                                                                                          
count: 1 , rewardCount: 500                                                                                             
count: 2 , rewardCount: 500                                                                                             
count: 3 , rewardCount: 500                                                                                             
count: 4 , rewardCount: 500                                                                                             
count: 5 , rewardCount: 500                                                                                             
count: 6 , rewardCount: 377                                                                                             
count: 7 , rewardCount: 405                                                                                             
count: 8 , rewardCount: 261                                                                                             
count: 9 , rewardCount: 500                                                                                             
count: 10 , rewardCount: 400                                                                                            
count: 11 , rewardCount: 500                                                                                            
count: 12 , rewardCount: 500                                                                                            
count: 13 , rewardCount: 474                                                                                            
count: 14 , rewardCount: 500                                                                                            
count: 15 , rewardCount: 442                                                                                            
count: 16 , rewardCount: 243                                                                                            
count: 17 , rewardCount: 409                                                                                            
count: 18 , rewardCount: 362                                                                                            
count: 19 , rewardCount: 500                                                                                            
count: 20 , rewardCount: 500                                                                                            
count: 21 , rewardCount: 500                                                                                            
count: 22 , rewardCount: 368                                                                                            
count: 23 , rewardCount: 500                                                                                            
count: 24 , rewardCount: 500                                                                                            
count: 25 , rewardCount: 500                                                                                            
count: 26 , rewardCount: 308          
****/
```



# DoubleDQN

DoubleDQN 是 DQN 的改进算法版，网络架构和 DQN 完全一样，**只修改 TD‑target 计算逻辑** 

DQN 损失函数 $\mathcal L_{\text{MSE}} = \big(r +\gamma \underbrace{Q_{\theta'}(s',a')}_{DoubleDQN只修改这里} - Q_{\theta}(s,a)\big)^2$

**DoubleDQN 改进点:**

1. **在线网络**，负责**挑选最优动作** $a^*=\max \limits_{a} Q_{\theta}(s',a')$ 
2. **目标网络** ，使用最优动作 $a^* , Q_{\theta'}(s',a^*)$

DoubleDQN 损失函数 $\mathcal L_{\text{MSE}} = \big(r +\gamma {Q_{\theta'}(s',a^*)} - Q_{\theta}(s,a)\big)^2$

 **DoubleDQN** 改进点代码

```
        auto [_, idx] = m_Qnet->forward(s1).max(1); // max(): (Tensor values, Tensor indices)
        idx = idx.view({ -1,1 });
        q1 = m_TargetQnet->forward(s1).gather(1, idx);
```



DoubleDQN 和 DQN 代码整合到一起 



```
void DeepQNetwork::Update()
{

    ReplayBuffer dataTrain; 
    static int count = 1;

    auto samples = dataTrain.sample(m_batchsize);

    auto [s0, a, r, s1, done] = QwListToTensor(samples, m_device);

    auto q = m_Qnet->forward(s0);
    q = q.gather(1, a);

    torch::Tensor q1 ;
    if (m_bDoubleDQN)
    {
        auto [_, idx] = m_Qnet->forward(s1).max(1); // max(): (Tensor values, Tensor indices)
        idx = idx.view({ -1,1 });
        q1 = m_TargetQnet->forward(s1).gather(1, idx);

    }
    else
    {
        q1 = m_TargetQnet->forward(s1);
        auto [qv, _] = q1.max(1);
        q1 = qv.view({ -1,1 });
    }

    auto qtargets = r + m_dbGamma * q1 * (1 - done);

    auto mseloss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
    auto dqnloss = mseloss->forward(q, qtargets);
    m_pAdam->zero_grad();
    dqnloss.backward();
    m_pAdam->step();
    auto loss  = dqnloss.item<double>();
    if (count % 10 == 0)
    {
        SyncTargetNet();
        //cout << "dqnloss: " << loss << endl;
    }
    count++;


}

```



**DoubleDQN运行效果**

```
int main()
{
    DeepQNetwork  deepQN;
    //deepQN.Play(400);
    deepQN.DoubleDQN(400);

}
/**
Currently DoubleDQN                                                                                                     
GenerateTrainData .....                                                                                                 
train i: 80 / 400 , rewardCount: 11                                                                                     
train i: 100 / 400 , rewardCount: 49                                                                                    
train i: 120 / 400 , rewardCount: 470                                                                                   
train i: 120, break Generate Train ####                                                                                 

TestData .....                                                                                                          
count: 1 , rewardCount: 500                                                                                             
count: 2 , rewardCount: 500                                                                                             
count: 3 , rewardCount: 500                                                                                             
count: 4 , rewardCount: 299                                                                                             
count: 5 , rewardCount: 500                                                                                             
count: 6 , rewardCount: 500                                                                                             
count: 7 , rewardCount: 500                                                                                             
count: 8 , rewardCount: 500                                                                                             
count: 9 , rewardCount: 308                                                                                             
count: 10 , rewardCount: 338                                                                                            
count: 11 , rewardCount: 265                                                                                            
count: 12 , rewardCount: 500                                                                                            
count: 13 , rewardCount: 312                                                                                            
count: 14 , rewardCount: 500                                                                                            
count: 15 , rewardCount: 265                                                                                            
count: 16 , rewardCount: 294                                                                                            
count: 17 , rewardCount: 353                                                                                            
count: 18 , rewardCount: 500                                                                                            
count: 19 , rewardCount: 253                                                                                            
count: 20 , rewardCount: 410                                                                                            
count: 21 , rewardCount: 313                                                                                            
count: 22 , rewardCount: 500                                                                                            
count: 23 , rewardCount: 281                                                                                            
count: 24 , rewardCount: 254                                                                                            
count: 25 , rewardCount: 500                                                                                            
count: 26 , rewardCount: 500                                                                                            

**/
```



# 车杆(CartPole) 环境得分评测

| 评级           | 100 回合平均奖励 | 表现说明                         |
| ------------ | ---------- | ---------------------------- |
| 🔴很差         | 0～50       | 几乎无法维持平衡，杆子迅速倾倒，接近随机策略水平     |
| 🟠较差         | 50～150     | 可以短暂维持平衡，但极易倾倒，策略不稳定，波动大     |
| 🟡合格         | 150～350    | 能坚持较长时间，但频繁失败；尚未达到环境解决标准     |
| 🟢良好         | 350～474    | 大部分回合接近满分，偶尔提前结束；接近通关，但未达标   |
| ✅通关 (Solved) | ≥475       | 达到 Gymnasium 官方解决标准，整体策略稳定可靠 |
| ⭐满分          | =500       | 绝大多数回合跑满 500 步，杆子全程不倒，最优性能   |





# 代码下载

[强化学习-进阶篇示例代码资源](https://download.csdn.net/download/qq00769539/93326504?spm=1011.2124.3001.6210)


