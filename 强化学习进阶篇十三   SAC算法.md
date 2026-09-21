# SAC 算法思想来源

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

      

# 高斯分布（正态分布）

## 1. 定义

一维高斯分布记作：$\boldsymbol{z \sim \mathcal N(\mu,\sigma^2)}$

- $\mu$：**均值**，分布中心
- $\sigma$：**标准差**，控制分布"胖瘦"
- $\sigma^2$：方差

概率密度函数 PDF（Probability Density Function）：

$p(z)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)$

> **概率密度 $p(z)$ 不是概率！**
> 
> 连续随机变量单点概率为 0；
> $P(a<z<b)=\int_{a}^{b} p(z)\,dz$
> 面积才是概率。

---

## 2. 最大熵原理推导高斯分布

高斯分布是**已知均值和方差时熵最大**的分布。

- 离散：$H(p)=-\sum_i p_i\ln p_i$
- 连续：$H(p)=-\int_{-\infty}^{+\infty} p(x)\ln p(x)\,dx$

### 2.1 约束条件

$\begin{cases}
\max\limits_{p(x)} \quad H(p)=-\displaystyle\int_{-\infty}^{+\infty}p(x)\ln p(x)\,dx \\[6pt]
\text{归一化：}\quad \displaystyle\int_{-\infty}^{+\infty} p(x)\,dx=1 \\[6pt]
\text{均值为 }\mu：\quad \displaystyle\int_{-\infty}^{+\infty} x\,p(x)\,dx=\mu \\[6pt]
\text{方差为 }\sigma^2：\quad \displaystyle\int_{-\infty}^{+\infty}(x-\mu)^2\,p(x)\,dx=\sigma^2
\end{cases}$

### 2.2 拉格朗日乘子法

构造拉格朗日函数：

$\mathcal L=-\int p(x)\ln p(x)\,dx
+\lambda_1\underbrace{\left(1-\int p(x)\,dx\right)}_{\text{归一化}}
+\lambda_2\underbrace{\left(\mu-\int x\,p(x)\,dx\right)}_{\text{均值}}
+\lambda_3\underbrace{\left(\sigma^2-\int (x-\mu)^2 p(x)\,dx\right)}_{\text{方差}}$

### 2.3 求导

对 $p(x)$ 求导并令其为 0：$\frac{\delta\mathcal L}{\delta p(x)}=0$

逐项求导：

$\begin{cases}\begin{aligned}
&\textcircled{1}\quad \frac{\delta}{\delta p}\left(-\int p\ln p\,dx\right) = -(\ln p+1) \\
&\textcircled{2}\quad \frac{\delta}{\delta p}\left[\lambda_1\left(1-\int p\,dx\right)\right] = -\lambda_1 \\
&\textcircled{3}\quad \frac{\delta}{\delta p}\left[\lambda_2\left(\mu-\int x p\,dx\right)\right] = -\lambda_2 x \\
&\textcircled{4}\quad\frac{\delta}{\delta p}\left[\lambda_3\left(\sigma^2-\int (x-\mu)^2 p\,dx\right)\right] = -\lambda_3(x-\mu)^2
\end{aligned} \end{cases}$

合并：

$\begin{cases}-\ln p-1-\lambda_1-\lambda_2 x-\lambda_3(x-\mu)^2=0\\
\ln p(x)=-1-\lambda_1-\lambda_2 x-\lambda_3(x-\mu)^2\end{cases}$

### 2.4 确定 $\lambda_2=0$

令 $y=x-\mu$，即 $x=y+\mu$，代入：

$\ln p=-1-\lambda_1-\lambda_2(y+\mu)-\lambda_3 y^2=(-1-\lambda_1-\lambda_2\mu)-\lambda_2 y-\lambda_3 y^2$



$对y配方=\begin{cases} -\lambda_2 y-\lambda_3 y^2=-\lambda_3(\frac{\lambda_2}{\lambda_3}y+y^2)\\\\
 -\lambda_3(\frac{\lambda_2}{\lambda_3}y+y^2)=-\lambda_3(-\frac{\lambda_2^2}{4\lambda_3^2}+\underbrace{\frac{\lambda_2^2}{4\lambda_3^2}+ \frac{\lambda_2}{\lambda_3}y  +y^2}_{=(\frac{\lambda_2}{2\lambda_3}+y)^2})\\\\
 -\lambda_2 y-\lambda_3 y^2=-\lambda_3\left(y+\frac{\lambda_2}{2\lambda_3}\right)^2+\frac{\lambda_2^2}{4\lambda_3}\\
\end{cases}$

$y$ 的中心（均值）为：

$\mathbb{E}[y]=-\frac{\lambda_2}{2\lambda_3}$

由均值约束 $\mathbb{E}[x]=\mu$，即 $\mathbb{E}[y]=0$：

$-\frac{\lambda_2}{2\lambda_3}=0 \quad\Longrightarrow\quad \boxed{\lambda_2=0}$

> **直观理解**：$\lambda_2$ 负责把高斯中心从 $\mu$ 推开；均值约束要求中心必须在 $\mu$，所以 $\lambda_2$ 只能是 0。

### 2.5 分布形式

$\begin{cases}\ln p(x)=-1-\lambda_1-\lambda_3(x-\mu)^2\\
令 C=-1-\lambda_1：\\
p(x)=e^{C}\cdot e^{-\lambda_3(x-\mu)^2}\end{cases}$

---

## 3. 求解拉格朗日乘子

### 3.1 高斯积分公式

$I=\int_{-\infty}^{\infty}e^{-ax^2}\,dx=\sqrt{\frac{\pi}{a}}$

由极坐标推导：

$I^2=\left(\int_{-\infty}^{\infty}e^{-ax^2}dx\right)\left(\int_{-\infty}^{\infty}e^{-ay^2}dy\right)=\int\!\!\int e^{-a(x^2+y^2)}dx\,dy=\int_0^{2\pi}\!\!\int_0^\infty e^{-ar^2}r\,dr\,d\theta=\frac{\pi}{a}$

**二阶矩**（对参数 $a$ 求导）：

$\frac{dI}{da}=\sqrt{\pi}\cdot\left(-\frac12\right)a^{-3/2}=-\frac12\sqrt{\pi}\,a^{-3/2}$

$\frac{dI}{da}=\int_{-\infty}^{\infty}\frac{\partial}{\partial a}e^{-ax^2}dx=-\int_{-\infty}^{\infty}x^2 e^{-ax^2}dx$

$\boxed{\int_{-\infty}^{\infty}x^2 e^{-ax^2}dx=\frac12\sqrt{\pi}\,a^{-3/2}}$

### 3.2 归一化方程

$式(1)=\begin{cases}\  \int_{-\infty}^{\infty}p(x)\,dx=1 \quad , p(x)=e^{C}\cdot e^{-\lambda_3(x-\mu)^2}\\
\ \int_{-\infty}^{\infty}e^{C}\cdot e^{-\lambda_3(x-\mu)^2}dx=e^{C}\cdot\sqrt{\frac{\pi}{\lambda_3}}=1\\
\ \boxed {e^{C}\cdot\sqrt{\pi}\cdot\lambda_3^{-1/2}=1}\\
\end{cases}$

### 3.3 方差方程

$式(2)=\begin{cases} \ \int_{-\infty}^{\infty}(x-\mu)^2 p(x)\,dx=\sigma^2 \quad , p(x)=e^{C}\cdot e^{-\lambda_3(x-\mu)^2} \\
\ \boxed {e^{C}\cdot\frac12\sqrt{\pi}\cdot\lambda_3^{-3/2}=\sigma^2} \end{cases}$

### 3.4 联立求解 $\lambda_3$

$式 (1) 和式 (2)解方程=\begin{cases} e^{C} \cdot \sqrt{\pi} \cdot \lambda_3^{-\frac{1}{2}}=1 \\e^{C} \cdot \dfrac12 \sqrt{\pi}\cdot \lambda_3^{-\frac32} =\sigma^2 \iff e^{C} \cdot \dfrac{1}{2\sigma^2} \sqrt{\pi}\cdot \lambda_3^{-\frac32} =1 \\\lambda_3^{-\frac{1}{2}}=\dfrac{1}{2\sigma^2} \cdot \lambda_3^{-\frac32} \iff \lambda_3=\dfrac{1}{2\sigma^2}\end{cases}$





$\boxed{\lambda_3=\frac{1}{2\sigma^2}}$

### 3.5 求解 $\lambda_1$

$\lambda_1 = \begin{cases} \textcircled{1} \quad C=-1 - \lambda_1 \\\textcircled{2} \quad \lambda_3=\dfrac{1}{2\sigma^2} 带入 \sqrt{\dfrac{\pi}{\lambda_3}} \quad \Rightarrow \sigma \sqrt{2\pi} \\\textcircled{3} \quad e^{(-1-\lambda_1)} \cdot\underbrace {\sigma \sqrt{2\pi}}_{= e^{\ln(\sigma \sqrt{2\pi}) }}=e^0 \quad \Rightarrow -1-\lambda_1 + \ln(\sigma \sqrt{2\pi})=0\\\textcircled{4} \quad \lambda_1=-1 + \ln(\sigma \sqrt{2\pi})\\\textcircled{5} \quad \lambda_1= \frac12\ln({2\pi})+\ln\sigma-1\\\end{cases}$



$\begin{cases}\quad \boxed{\lambda_1=\ln(\sigma\sqrt{2\pi})-1}\\\\
\quad\boxed{\lambda_1=\frac12\ln(2\pi)+\ln\sigma-1}\end{cases}$

---

## 4. 最终结果

### 拉格朗日乘子汇总

$\boxed{
\begin{aligned}
\lambda_1 &= \frac12\ln(2\pi)+\ln\sigma-1 \\
\lambda_2 &= 0 \\
\lambda_3 &= \frac{1}{2\sigma^2}
\end{aligned}
}$

### 对数概率密度

$\begin{cases}\ln p(x)=-1 - \lambda_1 - \lambda_3 \ (x-\mu)^2 \Rightarrow \\\\\ln p(x)=\begin{cases} -1-\lambda_1=-1-[-1 + \ln(\sigma \sqrt{2\pi})]= - \ln(\sigma \sqrt{2\pi}) \\ -\lambda_3 (x-\mu)^2=-\dfrac{(x-\mu)^2}{2\sigma^2}\\\end{cases} \\\\\boxed{\ln p(x)=-\ln(\sigma \sqrt{2\pi})-\dfrac{(x-\mu)^2}{2\sigma^2}=-\dfrac12 \ln(2\pi)-\ln \sigma - \dfrac12 (\dfrac{x-\mu}{\sigma})^2}\\\\ \end{cases}$

## 5.重参数化采样

从高斯分布 $\mathcal N(\mu,\sigma^2)$采样，并且采样结果可以对 $\mu,\sigma$ 求梯度。

$z = \mu + \varepsilon \cdot \sigma,\quad \varepsilon \sim \mathcal N(0,1)$

## 6.Tanh 压缩

- 高斯分布 $\mathcal N(\mu,\sigma^2)$ 的取值范围是 $(-\infty,+\infty)$

- $u = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

- $\tanh$ 的输出范围永远在 $(-1,1)$

## 7. 修正概率密度

核心问题：变换变量后，概率密度会变，也就说经过$\tanh$ 压缩概率密变了。

$z\sim U(0,2)$（0 到 2 均匀分布），质量计算公式$ \ 密度=\frac{质量}{体积}$  密度 $p_z(z)=\dfrac{1}{2-0}=0.5$

做变换 $u = \frac{1}{2}z$（把区间**压缩**到 0 到 1）概率密度变成了多少$p_u(u)=?$

答案：$p_u(u)=\dfrac{1}{1-0}=1$

### 7.1 变量替换法则

   如果 $u = g(z)$，单调可导，那么：$p_u(u) = p_z(z) \cdot \left|\frac{dz}{du}\right|$
  上面例子$u = g(z)=\frac12z \ , \quad p_z(z)=0.5$

- $\frac{dz}{du}$表示: $z=2u$函数对 $u$ 求导 所以$\frac{dz}{du}=2$
   答案：$p_u(u) = p_z(z) \cdot \left|\frac{dz}{du}\right|=0.5\cdot2=1$

- $\dfrac{dz}{du}=\dfrac{1}{\dfrac{du}{dz}},\dfrac{du}{dz}=g(z)'=\dfrac{1}{2}, \dfrac{dz}{du}=2$
  答案：$p_u(u) = p_z(z) \cdot \left|\frac{dz}{du}\right|=0.5\cdot2=1$

### 7.2 在SAC上的使用

1. $u = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

2. $u = \tanh(z) \frac{du}{dz} = 1-\tanh^2(z) = 1-u^2$

3. $p_u(u) = p_z(z) \cdot \left|\frac{dz}{du}\right|=p_z(z) \cdot \dfrac{1}{1-u^2}$
   
   

## 8. 正态分布的实现

$\boxed{\ln p(x)=-\ln(\sigma \sqrt{2\pi})-\dfrac{(x-\mu)^2}{2\sigma^2}=-\dfrac12 \ln(2\pi)-\ln \sigma - \dfrac12 (\dfrac{x-\mu}{\sigma})^2}$

```cpp
class NormalDistribution
{
public:
    NormalDistribution(torch::Tensor mean, torch::Tensor std)
        : m_mean(std::move(mean)),m_std(std::move(std))
    {
    }

    // 对应 Python：dist.rsample()
    torch::Tensor rsample() const
    {
        const auto epsilon = torch::randn_like(m_std);
        return m_mean + m_std * epsilon;
    }

    // 对应 上面公式
    torch::Tensor log_prob(const torch::Tensor& value) const
    {
        double logTwoPi = std::log(2 * M_PI);///1.8378770664093453;//

        return -0.5 * ((value - m_mean) / m_std).pow(2)- torch::log(m_std)- 0.5 * logTwoPi;
    }

private:
    torch::Tensor m_mean; // 均值
    torch::Tensor m_std;  // 方差
};

```

## 9. 随机策略网络(熵正则项)

核心思路：神经网络接收状态作为输入，输出高斯分布的均值与方差；基于该高斯分布采样得到原始动作，再经 Tanh 映射得到最终动作，同时计算动作对应的对数概率。

```cpp
class SACPolicyNetContImpl : public torch::nn::Module
{
public:

    SACPolicyNetContImpl() = default;
    SACPolicyNetContImpl(int64_t input, int64_t output, double actionBound, int64_t hidden = 128)
    {
        m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
        m_mu = register_module("mu", torch::nn::Linear(hidden, output));
        m_std = register_module("std", torch::nn::Linear(hidden, output));
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

        logProb = logProb - torch::log(1.0 - action.pow(2) + 1e-7);
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



# SAC 算法的实现

最难部分在最大熵原理推导过程,策略熵越大，动作分布越分散，探索能力越强；策略熵越小，动作越集中，更倾向于利用当前最优动作。

连续动作 SAC 主要包括：

- 一个随机 Actor（熵正则项）；
- 两个在线 Critic；
- 两个目标 Critic；
- 经验回放；
- 可自动训练的温度系数 $\alpha$；
- 目标 Critic 软更新。

SAC 使用两个 Critic 并取较小值，降低单个 Critic 对动作价值的过高估计。



### 1. Soft Critic 目标

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



### 2. Actor 目标

Actor 希望生成高价值动作，同时保持策略熵：

$\mathcal L_{Actor}=\mathbb E_{s\sim D,a\sim\pi_\theta}\left[\alpha\log\pi_\theta(a|s)-\min(Q_{\omega_1}(s,a),Q_{\omega_2}(s,a))\right]$

最小化该损失会：

- 增大两个 Critic 中较小的 $Q$ 值；
- 通过 $\alpha\log\pi(a|s)$ 保留策略随机性。
  
  

### 3. 创建 Actor、双 Critic 和目标 Critic

`void SAC::GenerateTrainData(int maxCount)`

```cpp
   cout << "Currently SAC (continuous)" << endl;
   m_maxMewardCount = 200;
   m_minLogCount = 20;
   m_minLogStep = 6;

   // 超参数（可按需要调整）
   m_dbGamma = 0.98;
   m_dbTau = 0.005;
   m_batchSize = 64;

   GetReplayDataList().clear();

   auto input = m_objEnv->GetStateDim();
   auto output = m_objEnv->GetActionDim();
   auto actionBound = m_objEnv->GetActionHigh();

   TORCH_CHECK(output == 1, "SAC currently supports one-dimensional continuous actions");

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

   // 将目标网络初始化为 critic 网络参数
   CopyModuleParameters(*m_critic1, *m_targetCritic1);
   CopyModuleParameters(*m_critic2, *m_targetCritic2);

   // 优化器
   m_pActorOpt = std::make_unique<torch::optim::Adam>(m_actor->parameters(), torch::optim::AdamOptions(m_dbActorLRDefault));
   m_pCritic1Opt = std::make_unique<torch::optim::Adam>(m_critic1->parameters(), torch::optim::AdamOptions(m_dbCriticLRDefault));
   m_pCritic2Opt = std::make_unique<torch::optim::Adam>(m_critic2->parameters(), torch::optim::AdamOptions(m_dbCriticLRDefault));

   // 可训练 log alpha (初始化为 log(0.01))
   m_logAlpha = torch::full({}, std::log(0.01), torch::TensorOptions().device(m_device).dtype(torch::kFloat32));
   m_logAlpha.set_requires_grad(true);
   // alpha 优化器，使用单张 tensor 参数列表
   m_pAlphaOpt = std::make_unique<torch::optim::Adam>(std::initializer_list<torch::Tensor>{m_logAlpha}, torch::optim::AdamOptions(m_dbAlphaLRDefault));

   // 训练模式
   m_actor->train();
   m_critic1->train();
   m_critic2->train();
   m_targetCritic1->eval();
   m_targetCritic2->eval();

   // 连续动作 SAC 的目标熵通常为 -动作维度
   m_dbTargetEntropy = -static_cast<double>(m_objEnv->GetActionDim());

   // 使用 BaseAdvanced 统一的数据生成循环（内部会调用 TrainGenerateItem1/2）
   BaseAdvanced::GenerateTrainData(maxCount);

   // eval 模式
   m_actor->eval();
   m_critic1->eval();
   m_critic2->eval();

   // 释放资源（unique_ptr 会自动释放）
   m_pActorOpt.reset();
   m_pCritic1Opt.reset();
   m_pCritic2Opt.reset();
   m_pAlphaOpt.reset();
```





### 4. 训练更新

`void SAC::TrainGenerateItem1(const QwItem& item)`

```cpp
 ReplayBuffer replayBuffer;
 auto samples = replayBuffer.sample(m_batchSize);

 if (samples.empty())
 {
     return;
 }

 auto [s0, a, reward, s1, done] = QwListToTensor(samples, m_device, true);

 torch::Tensor tdTarget;

 
 {
     torch::NoGradGuard noGrad;
     const auto alpha = m_logAlpha.exp();
     auto [nextAction, nextLogProb] = m_actor->forward(s1);
     const auto minTargetQ =torch::min(m_targetCritic1->forward(s1, nextAction), m_targetCritic2->forward(s1, nextAction));

     const auto nextValue =minTargetQ - alpha * nextLogProb;

     tdTarget =reward+ m_dbGamma * (1.0 - done) * nextValue;
 }

 // 更新两个Q网络
 {
     auto criticLoss1 = torch::mean(torch::mse_loss(m_critic1->forward(s0, a), tdTarget.detach()));
     auto criticLoss2 = torch::mean(torch::mse_loss(m_critic2->forward(s0, a), tdTarget.detach()));
     m_pCritic1Opt->zero_grad();
     m_pCritic2Opt->zero_grad();
     criticLoss1.backward();
     criticLoss2.backward();
     m_pCritic1Opt->step();
     m_pCritic2Opt->step();
 }
 // 更新策略网络

 torch::Tensor detachedLogProb;
 {
     auto [newAction, logProb] = m_actor->forward(s0);
     logProb = -logProb;
     detachedLogProb = logProb.detach();

		auto q1 = m_critic1->forward(s0, newAction);
		auto q2 = m_critic2->forward(s0, newAction);
     auto actorLoss = torch::mean(-m_logAlpha.exp() * logProb - torch::min(q1,q2));
		m_pActorOpt->zero_grad();
     actorLoss.backward();
		m_pActorOpt->step();
 }
 //更新alpha值
 {
 
     auto alphaLoss = torch::mean((detachedLogProb - m_dbTargetEntropy).detach() * m_logAlpha.exp());
     m_pAlphaOpt->zero_grad();
     alphaLoss.backward();
		m_pAlphaOpt->step();
 }


 SoftUpdate(*m_critic1, *m_targetCritic1);
 SoftUpdate(*m_critic2, *m_targetCritic2);
```








