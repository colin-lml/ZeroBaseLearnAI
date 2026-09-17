# SAC 算法

## 物理来源

玻尔兹曼分布是统计物理：热平衡下，粒子在不同能级的概率分布；能量越高，占据概率越低。RL 相当于把**能量换成动作价值 Q**。



## 玻尔兹曼分布定义

系统处于状态$i$ 的概率：

$p_i = \frac{e^{-E_i/(kT)}}{\sum_j e^{-E_j/(kT)}}$

* $E_i$：第 $i$ 个状态能量
* $k$：玻尔兹曼常数
* $T$：温度 \($T>0$\)
  
  

### RL 简化写法（动作选择，最常用）

把能量$E_i$替换成动作价值 $Q(s,a)$，引入温度参数 $\tau$：

$\pi(a|s)=\frac{\exp\left(Q(s,a)/\tau\right)}{\sum_{a'}\exp\left(Q(s,a')/\tau\right)}$

这个就是 **softmax 策略 = 玻尔兹曼探索**

### 简单例子

动作 $A：Q=5$；动作 $B：Q=3；\tau=2$

$\pi(A)=\frac{e^{5/2}}{e^{5/2}+e^{3/2}} \approx 0.731,\quad \pi(B)\approx0.269$

调高温度$\tau=10$：

$\pi(A)=\frac{e^{0.5}}{e^{0.5}+e^{0.3}}\approx0.550,\quad \pi(B)\approx0.450$

温度升高，两个动作概率差距变小。

## 正则化

**正则化是机器学习中防止模型过拟合、提升泛化能力的技术，通过在损失函数中添加惩罚项来限制模型复杂度**。它本质上是结构风险最小化策略的实现，旨在平衡训练误差与模型复杂度 。

- 传统动作价值函数 $Q(s_t,a_t)=r_t+ \gamma \mathbb{E}_{t+1}  [V(s_{t+1})]$ 则状态价值函数$V(s_t)=\mathbb{E}_{t}[Q(s_t,a_t)]$
- 熵正则化 则状态价值函数 $V(s_t)=\mathbb{E}_{t}[Q(s_t,a_t)] + \underbrace {\alpha \mathcal H(\dots)}_{熵正则项}$ ，其中$\alpha$是一个正则化的系数
- 熵正则项$\mathcal H(\dots)$ 用策略网络**逼近理论玻尔兹曼策略，本章没实现玻尔兹曼分布**，而是用**高斯分布** 它适用连续控制任务。

## 高斯分布（正态分布）

1. 定义
   一维高斯分布记作：$\boldsymbol{z \sim \mathcal N(\mu,\sigma^2)}$
   
   - $\mu$：**均值**，分布中心；
   
   - $\sigma$：**标准差**，控制分布 “胖瘦”；
   
   - $\sigma^2$：方差。
   
   概率密度函数 PDF（Probability Density Function）：
   $p(z)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)$
   
   > 概率密度 \(p(z)\)：不是概率！
   > 连续随机变量，单点概率为 0；
   > $\displaystyle P(a<z<b)=\int_{a}^{b} p(z)dz$，面积才是概率。

2. 概率密度函数 PDF推导过程，
   
   - **构造目标函数： 高斯分布是熵最大**的分布 $\max_{p(x)} \quad H(p)=-\int_{-\infty}^{+\infty}p(x)\ln p(x)\,dx$  ，连续用$\int$表示求和 离散用$\sum$ 表示求和
   
   - **约束条件** $\begin{cases}\max_{p(x)} \quad H(p)=-\int_{-\infty}^{+\infty}p(x)\ln p(x)\,dx \\ 
      归一化： \int_{-\infty}^{+\infty}\ p(x)dx=1 \quad 所有概率之和为1  \\
     均值为\mu：  \int_{-\infty}^{+\infty}\ x \ p(x) dx=\mu\\
     方差为 \sigma^2：  \int_{-\infty}^{+\infty}\ x^2\  p(x) dx=\sigma^2\\
     \end{cases}$
   
   - **拉格朗日乘子法**构造目标 $\mathcal L=-\int p(x)\ln p(x)\,dx + \lambda_1 \underbrace{\left(1-\int p(x)\,dx\right)}_{归一化}+\lambda_2 \underbrace{\left(\mu-\int x p(x)\,dx\right)}_{均值}+\lambda_3 \underbrace{\left(\sigma^2-\int x^2 p(x)\,dx\right)}_{方差为}$
   
   - **令$p=p(x)$ ，求偏导等于0** $\begin{cases} \\ \dfrac{\partial \mathcal L}{\partial p}= - \underbrace{\dfrac{\partial }{\partial p} \left(\int p\ln p \ dx \right)}_{= -(\ln p +1)} + \underbrace{\dfrac{\partial }{\partial p} \lambda_1 \left(1-\int p\,dx\right)}_{= -\lambda_1} + \underbrace{\dfrac{\partial }{\partial p} \lambda_2 \left(\mu-\int x \ p\,dx\right)}_{=-\lambda_2\ x} + \underbrace{\dfrac{\partial }{\partial p} \lambda_3 \left(\sigma^2-\int x^2 p\,dx\right)}_{=-\lambda_3 \ x}=0 \\ \\
     \dfrac{\partial \mathcal L}{\partial p}= -\ln p -1 - \lambda_1 - \lambda_2 \ x - \lambda_3 \ x^2=0 \iff \ln p=-1 - \lambda_1 -\lambda_2 \ x - \lambda_3 \ x^2 \\\\
     \ln p(x)=-1 - \lambda_1 -\lambda_2 \ x - \lambda_3 \ x^2 \iff p(x)=e^{(-1 - \lambda_1)+ (-\lambda_2 \ x)+ (- \lambda_3 \ x^2)} \\\\
     \dfrac{\partial \mathcal L}{\partial \lambda_1}=1 -\int p \ dx=0 \quad  归一化\\\\
     \dfrac{\partial \mathcal L}{\partial \lambda_2}=\mu -\int x p \ dx=0 \quad 均值\\\\
     \dfrac{\partial \mathcal L}{\partial \lambda_3}=\sigma^2 -\int x^2 p \ dx=0 \quad 方差\\\\
     \end{cases}$
   
   - $求解 \lambda_1、\lambda_2、\lambda_3=\begin{cases}均值为=\mu \quad \lambda_2 \left(\mu-\int x p(x)\,dx\right)=0  \\\\ 
     高斯积分公式  I=\int e^{-ax^2}dx=\sqrt{\frac{\pi}{a}} \quad 由极坐标推导出来 I^2 = \left(\int_{-\infty}^{\infty} e^{-a x^2}dx\right)\left(\int_{-\infty}^{\infty} e^{-a y^2}dy\right) \\\\
     \textcircled{1} =\dfrac{dI}{d(a)}=\dfrac{dI}{d(a)}\left(\sqrt{\frac{\pi}{a}}=\sqrt{\pi}\cdot a^{-\frac{1}{2}}\right)=-\dfrac12 \sqrt{\pi}\cdot a^{-\frac12-1}=-\dfrac12 \sqrt{\pi}\cdot a^{-\frac32}\\\\
      \textcircled{2} =\dfrac{dI}{d(a)}=\int e^{-ax^2}dx= \int \frac{\partial}{\partial \lambda_3} e^{-ax^2}dx=\int -x^2 e^{-ax^2}dx =-\int x^2 e^{-ax^2}dx\\\\
       \textcircled{3} =\dfrac{dI}{d(a)}=\int e^{-ax^2}dx= \underbrace{\int x^2 e^{-ax^2}dx=  \dfrac12 \sqrt{\pi}\cdot a^{-\frac32}}\\\\
     归一化=1\quad \int p(x)\ dx=1 \iff \int e^{-1 - \lambda_1} \cdot  e^{-\lambda_3 \ x^2}\ dx= e^{-1 - \lambda_1} \cdot \sqrt{\dfrac{\pi}{\lambda_3}}=e^{-1 - \lambda_1} \cdot \sqrt{\pi} \cdot \lambda_3^{-\frac{1}{2}}=1\\\\
     方差为 =\sigma^2\quad \int x^2 p(x)dx=\sigma^2 \iff \int x^2 e^{-1 - \lambda_1} \cdot e^{-\lambda_3 \ x^2}\ dx =e^{-1 - \lambda_1} \cdot \dfrac12 \sqrt{\pi}\cdot \lambda_3^{-\frac32} =\sigma^2\\\\
     解方程=\begin{cases} e^{-1 - \lambda_1} \cdot \sqrt{\pi} \cdot \lambda_3^{-\frac{1}{2}}=1 \\
     e^{-1 - \lambda_1} \cdot \dfrac12 \sqrt{\pi}\cdot \lambda_3^{-\frac32} =1 \end{cases}\\\\
     \lambda_3=\dfrac12\\\\
     \lambda_1=\dfrac12 \ln(2\pi)-1\\\\
     p(x)=e^{(-1 - \lambda_1)+ (- \lambda_3 \ x^2)}=e^{-1 -(\frac12 \ln(2\pi)-1 ) } \cdot e^{-\frac12 x^2}= \underbrace{e^{-\frac12 \ln(2\pi)}}_{k \ln         a=\ln a^{k}\quad y=e^{ln(y)}} \cdot e^{-\frac12 x^2}\\\\
     p(x)=\dfrac{1}{\sqrt{2\pi}} \cdot e^{-\frac12 x^2}
     \end{cases}$

       







## SAC 算法由来

DDPG 使用确定性 Actor 直接输出连续动作：

$a=\mu_\theta(s)$

训练时需要额外加入高斯噪声进行探索。确定性策略对噪声大小、Critic 的估计误差和网络初始值比较敏感，训练过程可能不稳定。

SAC（Soft Actor-Critic，柔性演员-评论家）使用**随机策略**和**最大熵强化学习**。它不仅希望获得较高的累计奖励，还希望策略保持一定的随机性：

$J(\pi)=\mathbb E_\pi\left[\sum_t\gamma^t\left(r_t+\alpha\mathcal H(\pi(\cdot|s_t))\right)\right]$

其中：

- $r_t$：环境奖励；
- $\mathcal H(\pi(\cdot|s_t))$：策略熵；
- $\alpha$：温度系数，用于平衡奖励与熵。

策略熵越大，动作分布越分散，探索能力越强；策略熵越小，动作越集中，更倾向于利用当前最优动作。

连续动作 SAC 主要包括：

- 一个随机 Actor；
- 两个在线 Critic；
- 两个目标 Critic；
- 经验回放；
- 可自动训练的温度系数 $\alpha$；
- 目标 Critic 软更新。

SAC 使用两个 Critic 并取较小值，降低单个 Critic 对动作价值的过高估计。



## SAC 公式推导

### 1. 最大熵目标

策略熵可以写为：

$\mathcal H(\pi(\cdot|s))=\mathbb E_{a\sim\pi}\left[-\log\pi(a|s)\right]$

因此最大熵目标同时鼓励：

1. 获得更高的环境奖励；
2. 保持更高的动作随机性。

温度系数 $\alpha$ 决定熵的重要程度：

- $\alpha$ 较大：更重视探索；
- $\alpha$ 较小：更重视当前奖励；
- $\alpha\rightarrow0$：目标逐渐接近普通的奖励最大化。
  
  

### 2. Soft Critic 目标

Actor 在下一状态采样动作：

$a'\sim\pi_\theta(\cdot|s')$

使用两个目标 Critic 的较小值：

$Q_{min}'(s',a')=\min\left(Q_{\omega_1'}(s',a'),Q_{\omega_2'}(s',a')\right)$

Soft 状态价值目标为：

$V(s')=Q_{min}'(s',a')-\alpha\log\pi_\theta(a'|s')$

最终 TD 目标为：

$y=r+\gamma(1-done)\left(Q_{min}'(s',a')-\alpha\log\pi_\theta(a'|s')\right)$

两个 Critic 分别最小化：

$\mathcal L_{Q_i}=\mathbb E\left[\left(Q_{\omega_i}(s,a)-y\right)^2\right],\quad i\in\{1,2\}$



### 3. Actor 目标

Actor 希望生成高价值动作，同时保持策略熵：

$\mathcal L_{Actor}=\mathbb E_{s\sim D,a\sim\pi_\theta}\left[\alpha\log\pi_\theta(a|s)-\min(Q_{\omega_1}(s,a),Q_{\omega_2}(s,a))\right]$

最小化该损失会：

- 增大两个 Critic 中较小的 $Q$ 值；
- 通过 $\alpha\log\pi(a|s)$ 保留策略随机性。
  
  

### 4. 自动温度系数

固定的 $\alpha$ 需要针对不同环境手动调整。SAC 可以设置目标熵并自动训练温度系数：

$\alpha=\exp(\log\alpha)$

使用 $\log\alpha$ 作为可训练参数，可以保证：

$\alpha>0$

本实现将目标熵设置为负动作维度：

$\mathcal H_{target}=-|\mathcal A|$

当前环境只支持一维连续动作，因此：

$\mathcal H_{target}=-1$

**总结：** SAC 使用随机 Actor 提供探索，使用熵正则平衡探索与利用，使用双 Critic 减少价值过估计，并自动调整温度系数。



## SAC 实现细节

### 0. 正态分布

```
class NormalDistribution
{
public:
    NormalDistribution(torch::Tensor mean, torch::Tensor std)
        : m_mean(std::move(mean)), m_std(std::move(std))
    {
    }

    torch::Tensor rsample() const
    {
        const auto epsilon = torch::randn_like(m_std);
        return m_mean + m_std * epsilon;
    }

    torch::Tensor log_prob(const torch::Tensor& value) const
    {
        constexpr double logTwoPi = 1.8378770664093453;

        return -0.5 * ((value - m_mean) / m_std).pow(2)
            - torch::log(m_std)
            - 0.5 * logTwoPi;
    }

private:
    torch::Tensor m_mean;
    torch::Tensor m_std;
};
```

Actor 输出正态分布的均值和标准差，然后从分布中采样连续动作。

`rsample()` 使用重参数化技巧：

$u=\mu_\theta(s)+\sigma_\theta(s)\epsilon$

$\epsilon\sim\mathcal N(0,1)$

随机性来自与网络参数无关的 $\epsilon$，采样结果 $u$ 仍然可以对 $\mu_\theta$ 和 $\sigma_\theta$ 求导，使 Actor 能够通过 Critic 反向传播训练。

`log_prob()` 计算正态分布采样值的对数概率。



### 1. SAC 随机 Actor 网络

```
class SACPolicyNetContImpl : public torch::nn::Module
{
public:
    SACPolicyNetContImpl() = default;

    SACPolicyNetContImpl(
        int64_t input,
        int64_t output,
        double actionBound,
        int64_t hidden = 128)
    {
        m_fc1 = register_module(
            "fc1", torch::nn::Linear(input, hidden));
        m_mu = register_module(
            "mu", torch::nn::Linear(hidden, output));
        m_std = register_module(
            "std", torch::nn::Linear(hidden, output));
        m_dbActionBound = actionBound;
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
        auto mu = m_mu->forward(x);
        auto std = torch::softplus(m_std->forward(x)) + 1e-6;

        NormalDistribution normal(mu, std);
        auto normalSample = normal.rsample();
        auto logProb = normal.log_prob(normalSample);
        auto action = torch::tanh(normalSample);

        logProb = logProb
            - torch::log(1.0 - action.pow(2) + 1e-7);
        logProb = logProb.sum(-1, true);
        action = action * m_dbActionBound;

        return { action, logProb };
    }

    torch::Tensor mean_action(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
        return m_dbActionBound * torch::tanh(m_mu->forward(x));
    }

private:
    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_mu{ nullptr };
    torch::nn::Linear m_std{ nullptr };
    double m_dbActionBound = 2.0;
};

TORCH_MODULE(SACPolicyNetCont);
```

网络共享一个隐藏层，然后分成两个输出分支：

- `m_mu`：输出正态分布均值 $\mu_\theta(s)$；
- `m_std`：输出标准差对应的值。

使用 `softplus` 保证标准差为正：

$\sigma=\operatorname{softplus}(x)+10^{-6}$



### 2. Tanh 动作压缩与概率修正

正态分布的取值范围是 $(-\infty,+\infty)$，而环境动作有上下界。代码先采样未压缩变量 $u$，再使用：

$a=\tanh(u)$

将动作压缩到 $[-1,1]$，最后乘以 `actionBound` 得到环境动作。

由于 `tanh` 改变了概率密度，必须修正对数概率：

$\log\pi(a|s)=\log\mathcal N(u;\mu,\sigma)-\log(1-\tanh^2(u))$

代码中加入 $10^{-7}$，防止动作接近 $-1$ 或 $1$ 时计算 $\log(0)$：

```
logProb = logProb
    - torch::log(1.0 - action.pow(2) + 1e-7);
```

多维动作需要对所有动作维度的对数概率求和：

```
logProb = logProb.sum(-1, true);
```



### 3. 训练动作与评测动作

```
double SAC::TakeAction(VectorDouble& s0, bool bPredict)
{
    torch::NoGradGuard no_grad;
    auto s = VectorDoubleTensor(s0, m_device);
    torch::Tensor action;

    if (bPredict)
    {
        action = m_actor->mean_action(s);
    }
    else
    {
        auto result = m_actor->forward(s);
        action = std::get<0>(result);
    }

    auto value = action.squeeze().item<double>();
    return std::clamp(
        value,
        m_objEnv->GetActionLow(),
        m_objEnv->GetActionHigh());
}
```

训练时从 Actor 的正态分布中随机采样动作，因此 SAC 不需要像 DDPG 一样额外添加高斯探索噪声。

评测时使用均值分支：

$a=actionBound\cdot\tanh(\mu_\theta(s))$

这样可以获得稳定的确定性评测动作。最后使用 `std::clamp` 保证动作位于环境合法范围内。



### 4. 创建 Actor、双 Critic 和目标 Critic

```
m_actor = SACPolicyNetCont(input, output, actionBound);
m_critic1 = QValueNetCont(input, output);
m_critic2 = QValueNetCont(input, output);
m_targetCritic1 = QValueNetCont(input, output);
m_targetCritic2 = QValueNetCont(input, output);

m_actor->to(m_device);
m_critic1->to(m_device);
m_critic2->to(m_device);
m_targetCritic1->to(m_device);
m_targetCritic2->to(m_device);

CopyModuleParameters(*m_critic1, *m_targetCritic1);
CopyModuleParameters(*m_critic2, *m_targetCritic2);
```

SAC 创建五个网络：

1. 随机 Actor；
2. 在线 Critic 1；
3. 在线 Critic 2；
4. 目标 Critic 1；
5. 目标 Critic 2。

与 DDPG 不同，本实现没有目标 Actor。计算下一状态 TD 目标时，直接使用当前随机 Actor 采样动作，并通过 `NoGradGuard` 阻止梯度传播。

当前实现要求动作维度为 $1$：

```
TORCH_CHECK(
    output == 1,
    "SAC currently supports one-dimensional continuous actions");
```



### 5. 创建优化器和温度参数

```
m_pActorOpt = std::make_unique<torch::optim::Adam>(
    m_actor->parameters(),
    torch::optim::AdamOptions(m_dbActorLRDefault));
m_pCritic1Opt = std::make_unique<torch::optim::Adam>(
    m_critic1->parameters(),
    torch::optim::AdamOptions(m_dbCriticLRDefault));
m_pCritic2Opt = std::make_unique<torch::optim::Adam>(
    m_critic2->parameters(),
    torch::optim::AdamOptions(m_dbCriticLRDefault));

m_logAlpha = torch::full(
    {}, std::log(0.01),
    torch::TensorOptions()
        .device(m_device)
        .dtype(torch::kFloat32));
m_logAlpha.set_requires_grad(true);

m_pAlphaOpt = std::make_unique<torch::optim::Adam>(
    std::initializer_list<torch::Tensor>{ m_logAlpha },
    torch::optim::AdamOptions(m_dbAlphaLRDefault));
```

Actor、两个 Critic 和温度参数分别使用独立的 Adam 优化器，学习率均为 $10^{-3}$。

温度系数初始化为：

$\alpha=0.01$

实际训练参数是：

$\log\alpha=\log(0.01)$

使用时通过 `m_logAlpha.exp()` 得到始终为正的 $\alpha$。



### 6. 经验回放与训练时机

```
void SAC::TrainGenerateItem1(const QwItem& item)
{
    AddReplayDataList(item);

    if (GetReplayDataList().size() >
        static_cast<size_t>(m_nMinimalsize))
    {
        Update();
    }
}
```

每次与环境交互后，将：

$(s,a,r,s',done)$

加入经验回放缓冲区。当样本数量超过 500 后，每产生一个新样本就随机采样 64 条数据更新网络。

SAC 与 DDPG 一样属于离策略（Off-Policy）算法，可以使用历史策略产生的经验，提高样本利用率。



### 7. 计算 Soft TD 目标

```
torch::Tensor tdTarget;
{
    torch::NoGradGuard noGrad;
    const auto alpha = m_logAlpha.exp();
    auto [nextAction, nextLogProb] = m_actor->forward(s1);
    const auto minTargetQ = torch::min(
        m_targetCritic1->forward(s1, nextAction),
        m_targetCritic2->forward(s1, nextAction));

    const auto nextValue = minTargetQ - alpha * nextLogProb;

    tdTarget = reward
        + m_dbGamma * (1.0 - done) * nextValue;
}
```

Actor 在下一状态随机采样动作和动作对数概率：

$a'\sim\pi_\theta(\cdot|s')$

两个目标 Critic 取较小值：

$Q_{min}'=\min(Q_{\omega_1'}(s',a'),Q_{\omega_2'}(s',a'))$

加入熵奖励后的下一状态价值为：

$V(s')=Q_{min}'-\alpha\log\pi_\theta(a'|s')$

TD 目标为：

$y=r+\gamma(1-done)V(s')$

整个目标计算位于 `torch::NoGradGuard` 中，不更新 Actor、目标 Critic和温度参数。



### 8. 更新两个 Critic

```
auto criticLoss1 = torch::mean(torch::mse_loss(
    m_critic1->forward(s0, a), tdTarget.detach()));
auto criticLoss2 = torch::mean(torch::mse_loss(
    m_critic2->forward(s0, a), tdTarget.detach()));

m_pCritic1Opt->zero_grad();
m_pCritic2Opt->zero_grad();
criticLoss1.backward();
criticLoss2.backward();
m_pCritic1Opt->step();
m_pCritic2Opt->step();
```

两个 Critic 使用同一个 TD 目标，但参数独立：

$\mathcal L_{Q_1}=\operatorname{MSE}(Q_{\omega_1}(s,a),y)$

$\mathcal L_{Q_2}=\operatorname{MSE}(Q_{\omega_2}(s,a),y)$

后续计算目标和 Actor 损失时使用二者较小值，降低价值过估计造成的策略错误更新。



### 9. 更新随机 Actor

```
auto [newAction, logProb] = m_actor->forward(s0);
logProb = -logProb;
detachedLogProb = logProb.detach();

auto q1 = m_critic1->forward(s0, newAction);
auto q2 = m_critic2->forward(s0, newAction);
auto actorLoss = torch::mean(
    -m_logAlpha.exp() * logProb - torch::min(q1, q2));

m_pActorOpt->zero_grad();
actorLoss.backward();
m_pActorOpt->step();
```

网络原始输出的 `logProb` 是 $\log\pi(a|s)$。代码先执行：

$logProb\leftarrow-\log\pi(a|s)$

它可以看作当前采样动作的熵估计。代入代码中的损失：

$\mathcal L_{Actor}=\mathbb E\left[-\alpha(-\log\pi(a|s))-Q_{min}(s,a)\right]$

整理后得到标准 SAC Actor 损失：

$\mathcal L_{Actor}=\mathbb E\left[\alpha\log\pi(a|s)-Q_{min}(s,a)\right]$

Actor 通过重参数化采样接收梯度，同时学习提高动作价值并维持策略熵。



### 10. 更新温度系数

```
auto alphaLoss = torch::mean(
    (detachedLogProb - m_dbTargetEntropy).detach()
    * m_logAlpha.exp());

m_pAlphaOpt->zero_grad();
alphaLoss.backward();
m_pAlphaOpt->step();
```

代码中的 `detachedLogProb` 保存 $-\log\pi(a|s)$，温度损失为：

$\mathcal L_\alpha=\mathbb E\left[\alpha\left(-\log\pi(a|s)-\mathcal H_{target}\right)\right]$

Actor 的熵估计调用 `detach()`，因此温度更新不会反向修改 Actor。只有 `m_logAlpha` 通过独立优化器更新。

目标熵设置为：

```
m_dbTargetEntropy =
    -static_cast<double>(m_objEnv->GetActionDim());
```

温度系数根据当前策略熵和目标熵之间的关系自动变化，用于调节 Actor 损失与 Soft TD 目标中熵项的权重。



### 11. 软更新目标 Critic

```
void SAC::SoftUpdate(
    torch::nn::Module& source,
    torch::nn::Module& target)
{
    torch::NoGradGuard no_grad;
    auto srcParams = source.parameters();
    auto tgtParams = target.parameters();

    for (size_t i = 0; i < srcParams.size(); ++i)
    {
        tgtParams[i].mul_(1.0 - m_dbTau);
        tgtParams[i].add_(srcParams[i], m_dbTau);
    }
}
```

每次训练结束后执行：

```
SoftUpdate(*m_critic1, *m_targetCritic1);
SoftUpdate(*m_critic2, *m_targetCritic2);
```

更新公式为：

$\omega_i'\leftarrow(1-\tau)\omega_i'+\tau\omega_i$

本实现使用 $\tau=0.005$。目标 Critic 缓慢跟随在线 Critic，使 TD 目标保持稳定。



### 12. 完整训练流程

SAC 的一次更新流程为：

1. 从经验回放中随机采样 64 条数据；
2. Actor 在下一状态采样动作和对数概率；
3. 两个目标 Critic 计算较小的目标 $Q$ 值；
4. 加入策略熵项，构造 Soft TD 目标；
5. 分别更新两个在线 Critic；
6. Actor 重新采样当前状态动作；
7. 使用 $\alpha\log\pi-Q_{min}$ 更新 Actor；
8. 根据策略熵更新温度系数 $\alpha$；
9. 软更新两个目标 Critic；
10. 使用更新后的随机策略继续与环境交互。
    
    

## SAC 与 DDPG 的区别

| 项目        | DDPG          | SAC             |
| --------- | ------------- | --------------- |
| Actor 类型  | 确定性策略         | 随机策略            |
| 动作输出      | 直接输出动作        | 输出正态分布参数并采样     |
| 探索方式      | 额外加入高斯噪声      | 策略分布自然采样        |
| Critic 数量 | 一个            | 两个              |
| 目标 Critic | 一个            | 两个              |
| 目标 Actor  | 使用            | 不使用             |
| 价值目标      | 奖励与下一状态 $Q$ 值 | 奖励、下一状态 $Q$ 值和熵 |
| Actor 目标  | 最大化 $Q$ 值     | 最大化 $Q$ 值和策略熵   |
| 温度系数      | 无             | 自动训练 $\alpha$   |
| 目标网络更新    | 软更新           | 软更新             |
| 经验回放      | 使用            | 使用              |

DDPG 追求当前状态下的单个最优动作，SAC 学习一个既能获得高奖励又保持随机性的动作分布。双 Critic 和最大熵目标通常使 SAC 具有更好的探索能力与训练稳定性。
