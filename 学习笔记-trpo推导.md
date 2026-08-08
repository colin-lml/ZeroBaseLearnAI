# 1.原始定义

$\boldsymbol{L(\theta')=\mathbb{E}_{s\sim\rho_\theta,\,a\sim\pi_\theta}\left[\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)} \cdot A_\theta(s,a)\right]}$

符号说明

* $\pi_\theta$：**旧策略，固定不动**
* $\pi_{\theta'}$：新策略，变量，我们优化 $\theta'$
* $A_\theta(s,a)$：**优势函数 Advantage**
* $\dfrac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}$：重要性采样比率（probability ratio，PPO 里也用）
* 期望是用**旧策略采样出来的数据**，数据不重新采样

**最大化替代目标，同时约束新($\pi_{\theta'}$)旧($\pi_\theta$)策略 KL 散度不超过阈值**，避免策略一次性更新过大导致训练崩塌。





# 2.泰勒展开

令 $\theta'=\theta+\Delta\theta$，$\Delta\theta$ 是参数增量。

一阶泰勒：

$L(\theta+\Delta\theta)\approx \underbrace{L(\theta)}_{常量丢掉}+\underbrace{\nabla_{\theta}L(\theta)^\top}_{g=\nabla_{\theta}L(\theta)} \Delta\theta$



$L$**一阶展开** 优化简化为: $\max g^\top \Delta\theta$



# 3.KL 散度








