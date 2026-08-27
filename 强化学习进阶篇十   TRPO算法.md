# TRPO 算法

## TRPO 算法由来

ActorCritic 使用策略网络 Actor 选择动作，使用价值网络 Critic 计算 TD 误差，并根据下面的损失函数更新 Actor：

$$\mathcal L_{Actor}=-\mathbb E_t\left[\log\pi_\theta(a_t|s_t)A_t\right]$$

普通策略梯度使用学习率控制每次参数更新的大小，但是参数变化小并不一定表示策略概率分布变化小。一次过大的策略更新可能使原来表现较好的动作概率突然降低，导致新策略性能严重下降。

TRPO（Trust Region Policy Optimization，信赖域策略优化）限制新旧策略之间的 KL 散度，在一个可信赖的策略变化范围内尽可能提高策略目标：

$$\max_\theta\quad \mathbb E_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}A_t\right]$$

$$\text{s.t.}\quad \mathbb E_t\left[D_{KL}\left(\pi_{\theta_{old}}(\cdot|s_t)\|\pi_\theta(\cdot|s_t)\right)\right]\leq\delta$$

其中 $\delta$ 是 KL 散度上限。TRPO 的核心思想是：**策略可以更新，但每次不能离旧策略太远。**

TRPO 仍然采用 ActorCritic 结构：

- **Actor：** 输出动作概率，使用信赖域方法更新；
- **Critic：** 估计状态价值，使用 Adam 优化器更新；
- **GAE：** 根据 TD 误差计算优势函数；
- **KL 约束：** 限制新旧策略分布之间的差异。



## TRPO 公式推导

### 1. 重要性采样与代理目标

训练数据由旧策略 $\pi_{\theta_{old}}$ 采样得到，但是要评价新策略 $\pi_\theta$。使用重要性采样比率：

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

代理目标（Surrogate Objective）为：

$$L(\theta)=\mathbb E_t\left[r_t(\theta)A_t\right]$$

当新旧策略相同时，$r_t(\theta)=1$。如果某个动作的优势 $A_t>0$，增大该动作在新策略中的概率可以提高代理目标；如果 $A_t<0$，则应该减小该动作的概率。

为了提高数值稳定性，代码通过对数概率计算比率：

$$r_t(\theta)=\exp\left(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t)\right)$$



### 2. KL 散度约束

TRPO 使用 KL 散度衡量新旧策略之间的距离：

$$\bar D_{KL}(\theta_{old},\theta)=\mathbb E_t\left[D_{KL}\left(\pi_{\theta_{old}}(\cdot|s_t)\|\pi_\theta(\cdot|s_t)\right)\right]$$

策略更新需要满足：

$$\bar D_{KL}(\theta_{old},\theta)\leq\delta$$

本实现设置：

$$\delta=5\times10^{-4}$$

KL 约束直接限制动作概率分布的变化，比简单限制参数变化更符合策略优化的目标。



### 3. 二阶近似

直接求解带 KL 约束的神经网络优化问题非常困难。TRPO 在旧参数 $\theta_{old}$ 附近进行近似。

对代理目标进行一阶近似：

$$L(\theta_{old}+x)\approx L(\theta_{old})+g^Tx$$

其中：

$$g=\nabla_\theta L(\theta)\big|_{\theta=\theta_{old}}$$

对 KL 散度进行二阶近似：

$$\bar D_{KL}(\theta_{old},\theta_{old}+x)\approx\frac{1}{2}x^THx$$

其中 $H$ 是 KL 散度关于策略参数的 Hessian 矩阵。

近似后的优化问题为：

$$\max_x\quad g^Tx$$

$$\text{s.t.}\quad\frac{1}{2}x^THx\leq\delta$$

其搜索方向为：

$$d=H^{-1}g$$

满足 KL 约束的完整步长为：

$$x=\sqrt{\frac{2\delta}{d^THd}}d$$

实际实现不会直接构造和求逆巨大的 Hessian 矩阵，而是使用 Hessian 向量积和共轭梯度法近似求解 $H^{-1}g$。



### 4. GAE 优势函数

TRPO 使用 GAE（Generalized Advantage Estimation，广义优势估计）降低优势估计的方差。

首先计算单步 TD 误差：

$$\delta_t=r_t+\gamma V(s_{t+1})(1-done_t)-V(s_t)$$

然后从回合末尾向前递推：

$$A_t=\delta_t+\gamma\lambda A_{t+1}$$

展开后为：

$$A_t=\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots$$

其中：

- $\gamma$ 控制未来奖励的折扣；
- $\lambda$ 控制偏差与方差之间的平衡；
- 本实现设置 $\gamma=0.98$、$\lambda=0.95$。



## TRPO 实现细节

### 0. Actor 和 Critic 网络

TRPO 使用两个神经网络：

```
PolicyNet m_ActorNet;
ValueNet m_CriticNet;
```

Actor 使用 `PolicyNet`，输入状态，经过 `softmax` 输出所有离散动作的概率：

$$\pi_\theta(\cdot|s)=[P(a_0|s),P(a_1|s),\ldots]$$

Critic 使用 `ValueNet`，输入状态并输出一个标量状态价值：

$$V_\omega(s)$$

Actor 不使用普通 Adam 优化器，而是通过共轭梯度和回溯线搜索更新参数。Critic 仍然使用 Adam 优化器最小化价值损失。



### 1. 创建网络和 Critic 优化器

```
void TRPO::GenerateTrainData(int maxCount)
{
    cout << "Currently TRPO" << endl;

    m_dbGamma = 0.98;
    m_dbAlpha = 0.5;

    auto input = m_objEnv->GetStateDim();
    auto output = m_objEnv->GetActionDim();

    m_ActorNet = PolicyNet(input, output);
    m_CriticNet = ValueNet(input, 1);

    m_CriticNet->to(m_device);
    m_ActorNet->to(m_device);

    m_pAdamCritic = new torch::optim::Adam(
        m_CriticNet->parameters(), { m_dbCriticLR });

    m_ActorNet->train();
    m_CriticNet->train();

    BaseAdvanced::GenerateTrainData(maxCount);

    m_ActorNet->eval();
    m_CriticNet->eval();

    delete m_pAdamCritic;
    m_pAdamCritic = nullptr;
}
```

主要超参数为：

- 折扣因子 `m_dbGamma = 0.98`；
- GAE 参数 `m_dbLmbda = 0.95`；
- Critic 学习率 `m_dbCriticLR = 1e-2`；
- KL 约束 `m_dbklConstraint = 5e-4`；
- 回溯线搜索系数 `m_dbAlpha = 0.5`。



### 2. 根据 Actor 选择动作

```
double TRPO::TakeAction(VectorDouble& s0, bool bPredict)
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

训练时按照 Actor 输出的类别分布采样动作，评测时选择概率最大的动作。TRPO 属于同策略（On-Policy）算法，当前回合数据由更新前的旧策略生成，并在策略更新时立即使用。



### 3. 更新 Critic

```
auto [s0, a, r, s1, done] = QwListToTensor(vList, m_device);

auto v0 = m_CriticNet->forward(s0);
auto v1 = r + m_dbGamma * m_CriticNet->forward(s1) * (1 - done);
auto td = v1 - v0;

auto criticLoss = torch::mean(torch::mse_loss(v0, v1.detach()));
m_pAdamCritic->zero_grad();
criticLoss.backward();
m_pAdamCritic->step();
```

Critic 的 TD 目标为：

$$y_t=r_t+\gamma V_\omega(s_{t+1})(1-done_t)$$

Critic 损失为：

$$\mathcal L_{Critic}=\operatorname{MSE}(V_\omega(s_t),y_t)$$

`v1.detach()` 将 TD 目标作为固定标签，避免梯度通过下一状态价值传播。变量 `td` 保存 Critic 更新前计算的 TD 误差，后续用于计算 GAE 优势。



### 4. 计算 GAE

```
torch::Tensor TRPO::ComputeAdvantage(
    double gamma, double lmbda, torch::Tensor& td)
{
    auto device = td.device();
    td.detach_();
    td = td.cpu().contiguous();

    auto n = td.size(0);
    auto m = td.size(1);
    std::vector<float> advantages(static_cast<size_t>(n * m));

    for (int64_t col = 0; col < m; ++col)
    {
        double adv = 0.0;
        for (int64_t i = n - 1; i >= 0; --i)
        {
            double delta = (m == 1)
                ? td[i].item<double>()
                : td[i][col].item<double>();
            adv = gamma * lmbda * adv + delta;
            advantages[static_cast<size_t>(i * m + col)]
                = static_cast<float>(adv);
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto adv = torch::from_blob(
        advantages.data(), { n, m }, options).clone();
    return adv.to(device);
}
```

代码从轨迹末尾向前执行：

$$adv\leftarrow\delta_t+\gamma\lambda adv$$

`td.detach_()` 切断 TD 误差与 Critic 计算图的联系，因为优势值只作为更新 Actor 的固定权重。

计算过程放在 CPU 上完成，最终再把优势张量移动回原来的设备。



### 5. 优势归一化

```
auto adv = ComputeAdvantage(m_dbGamma, m_dbLmbda, td);

auto mean = adv.mean();
auto std = adv.std();
auto adv_norm = (adv - mean) / (std + 1e-8);
```

优势归一化公式为：

$$\hat A_t=\frac{A_t-\operatorname{mean}(A)}{\operatorname{std}(A)+10^{-8}}$$

归一化不会改变动作优势的相对大小，可以减少不同回合奖励尺度变化对策略更新的影响。分母加 $10^{-8}$ 用于防止标准差为 $0$ 时除零。



### 6. 保存旧策略信息

```
auto logProbs = torch::log(
    m_ActorNet->forward(s0).gather(1, a)).detach();
auto actionDists = Categorical(
    m_ActorNet->forward(s0).detach());
```

在更新 Actor 前，需要保存旧策略：

- `logProbs`：旧策略对轨迹实际动作的对数概率；
- `actionDists`：旧策略在每个状态下的完整动作分布。

实际动作的旧概率用于计算重要性采样比率，完整旧分布用于计算新旧策略之间的 KL 散度。调用 `detach()` 后，这些数据不会随着 Actor 参数更新而变化。



### 7. 计算代理目标

```
torch::Tensor TRPO::ComputeSurrogateObj(
    const torch::Tensor& s,
    const torch::Tensor& a,
    const torch::Tensor& adv,
    const torch::Tensor& oldLogProbs,
    PolicyNet& actorNet)
{
    auto probs = actorNet->forward(s).gather(1, a);
    auto logProbs = torch::log(probs);
    auto ratio = torch::exp(logProbs - oldLogProbs);
    return (ratio * adv).mean();
}
```

对应公式：

$$r_t(\theta)=\exp\left(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)\right)$$

$$L(\theta)=\operatorname{mean}\left(r_t(\theta)\hat A_t\right)$$

TRPO 的目标是最大化代理目标，因此后续沿其梯度方向更新，而不是像常规损失函数一样执行梯度下降。



### 8. Hessian 向量积

```
auto newDists = Categorical(m_ActorNet->forward(s));
auto kl = torch::mean(oldsDists.kl_divergence(newDists));

auto grads = torch::autograd::grad(
    { kl }, params, {}, true, true, true);
auto vectorGrad = torch::cat(flatGradParts);
auto klGradVectorProduct = torch::dot(vectorGrad, v);
auto grad2 = torch::autograd::grad(
    { klGradVectorProduct }, params, {}, true, false, true);
auto Hv = torch::cat(flat2Parts);

constexpr double damping = 0.1;
return Hv + damping * v;
```

首先对平均 KL 散度求一次梯度，并保留计算图；然后将 KL 梯度与向量 $v$ 做内积，再对结果求一次梯度，得到：

$$Hv=\nabla_\theta\left((\nabla_\theta D_{KL})^Tv\right)$$

这种方法不需要显式构造大小为“参数量 × 参数量”的 Hessian 矩阵。

返回结果中加入阻尼项：

$$Hv\leftarrow Hv+0.1v$$

阻尼能够改善数值稳定性，避免 Hessian 接近奇异时共轭梯度求解发生剧烈波动。



### 9. 共轭梯度法

```
torch::Tensor TRPO::ConjugateGradient(
    const torch::Tensor& objGrad,
    const torch::Tensor& s,
    const Categorical& oldsDists)
{
    auto x = torch::zeros_like(objGrad);
    auto r = objGrad.clone();
    auto p = r.clone();
    auto rdotr = torch::dot(r, r);

    for (int i = 0; i < 20; i++)
    {
        auto Hp = HessianMatrixVectorProduct(s, oldsDists, p);
        auto alpha = rdotr / torch::dot(p, Hp);
        x += alpha * p;
        r -= alpha * Hp;
        auto new_rdotr = torch::dot(r, r);

        if (new_rdotr.item<double>() < 1e-9)
        {
            break;
        }

        auto beta = new_rdotr / rdotr;
        p = r + beta * p;
        rdotr = new_rdotr;
    }

    return x;
}
```

共轭梯度法用于近似求解线性方程：

$$Hx=g$$

返回的 $x$ 近似为：

$$x\approx H^{-1}g$$

本实现最多迭代 20 次。当残差平方小于 $10^{-9}$ 时提前停止。



### 10. 计算完整更新步长

```
auto surrogateObj = ComputeSurrogateObj(
    s, a, adv, oldLogProbs, m_ActorNet);
auto grads = torch::autograd::grad(
    { surrogateObj }, m_ActorNet->parameters());

auto flatGrad = torch::cat(flat);
auto searchDirection = ConjugateGradient(
    flatGrad, s, oldsDists);
auto Hd = HessianMatrixVectorProduct(
    s, oldsDists, searchDirection);
auto stepScale = torch::sqrt(
    2 * m_dbklConstraint /
    torch::dot(searchDirection, Hd));
auto fullStep = (stepScale * searchDirection).detach();
```

首先计算代理目标梯度 $g$，然后使用共轭梯度法得到搜索方向：

$$d\approx H^{-1}g$$

再按照 KL 约束缩放搜索方向：

$$fullStep=\sqrt{\frac{2\delta}{d^THd}}d$$

`fullStep` 是二阶近似下能够满足 KL 约束的最大更新步长，但由于神经网络是非线性的，实际更新后仍可能违反 KL 约束，因此还需要回溯线搜索。



### 11. 回溯线搜索

```
for (int i = 0; i < 15; i++)
{
    auto coefficient = std::pow(m_dbAlpha, i);
    auto newParams = oldParam + coefficient * fullStep;

    VectorToParameters(newParams, *tmpActor);

    auto newActionDists = Categorical(tmpActor->forward(s));
    auto kl = torch::mean(
        oldsDists.kl_divergence(newActionDists));
    auto newSurrogate = ComputeSurrogateObj(
        s, a, adv, oldLogProbs, tmpActor);

    if (newSurrogate.item<double>() >
            oldSurrogate.item<double>() &&
        kl.item<double>() < m_dbklConstraint)
    {
        bUpdate = true;
        return newParams.detach();
    }
}
```

第 $i$ 次尝试的参数为：

$$\theta_{new}=\theta_{old}+\alpha^i fullStep$$

本实现中 $\alpha=0.5$，最多尝试 15 次。候选参数必须同时满足两个条件：

1. 新代理目标大于旧代理目标；
2. 新旧策略平均 KL 散度小于 `m_dbklConstraint`。

如果完整步长不满足条件，就依次尝试 $0.5$、$0.25$、$0.125$ 倍步长。如果所有候选参数都不满足条件，则保留旧参数，不执行本次 Actor 更新。

使用临时策略网络 `tmpActor` 测试候选参数，可以避免在确认步长有效之前修改正式 Actor。



### 12. 完整 Actor 更新

```
bool bUpdate;
auto newParams = LineSearch(
    s, a, adv, oldLogProbs, oldsDists,
    fullStep, bUpdate);

if (bUpdate)
{
    VectorToParameters(newParams, *m_ActorNet);
}
```

TRPO 的 Actor 更新流程为：

1. 计算代理目标梯度 $g$；
2. 使用 Hessian 向量积表示 KL 曲率；
3. 使用共轭梯度法近似计算 $H^{-1}g$；
4. 根据 KL 上限计算完整步长；
5. 使用回溯线搜索检查代理目标和真实 KL 散度；
6. 找到合格参数后更新 Actor，否则放弃本次更新。



### 13. 训练终止条件

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

当单回合步数超过 450 时，`count` 加 $1$；如果下一回合未达标，`count` 清零。连续 4 个回合超过 450 后结束训练。

**训练终止条件：** 达到最大迭代次数，或连续 4 个回合的步数超过 450。



## TRPO 与 ActorCritic 的区别

| 项目 | ActorCritic | TRPO |
| --- | --- | --- |
| Actor 目标 | $\log\pi(a|s)A$ | 重要性采样代理目标 |
| Actor 更新 | Adam 梯度更新 | 共轭梯度与回溯线搜索 |
| 策略变化限制 | 依赖学习率 | 显式限制 KL 散度 |
| 优势估计 | 单步 TD 误差 | GAE |
| Critic 更新 | MSE + Adam | MSE + Adam |
| 二阶信息 | 不使用 | 使用 KL Hessian 向量积 |
| 实现复杂度 | 较低 | 较高 |
| 更新稳定性 | 可能出现过大更新 | 信赖域内更新更加稳定 |

TRPO 通过 KL 散度信赖域解决了普通策略梯度更新步长难以控制的问题，但共轭梯度、二阶自动微分和回溯线搜索使实现较复杂、计算量较大。

后续 PPO 算法保留“限制新旧策略差异”的思想，使用裁剪代理目标代替 TRPO 的二阶约束优化，在实现和计算上更加简单。
