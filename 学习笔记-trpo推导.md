# 1.原始定义

$\underbrace{\pi_{\boldsymbol{\theta}}(a|s)}_{动作概率分布}= \underbrace{\mathrm{softmax}\Big(w_2\, \max\big(w_1 s + b_1,\,0\big)+b_2\Big)}_{策略神经网络},\quad \boldsymbol{\theta}= \underbrace {\{w_1,b_1,w_2,b_2\}}_{网络参数}$



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



# 3. $D_{KL}$ 散度

令 $\theta'=\theta+\Delta\theta$，$\Delta\theta$ 是参数增量，$\pi_{\theta}=\pi_{\theta}(a|s)$



$D_{KL}(\pi_{\theta}||\pi_{\theta'})=\sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta}}(a|s)- \sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta'}}(a|s)$



# 4.$D_{KL}$二阶泰勒展开：

$D_{KL}(\pi_{\theta}||\pi_{\theta'})={\sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta}}(a|s)}- \sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta'}}(a|s)$

$D_{KL}(\theta+\Delta\theta) \approx \underbrace{ D_{KL}(\theta)}_{项1} + \underbrace{\nabla_{\theta'} D_{KL}(\theta)^\top \Delta\theta}_{项2}+ \underbrace{\dfrac{1}{2}\Delta\theta^\top H \Delta\theta}_{项3}$



**项1 代入 $\theta'=\theta$：** 

$D_{KL}(\theta)={\sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta}}(a|s)}- \underbrace{\sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta'}}(a|s)}_{\theta'=\theta}=0$

**项2 代入 $\theta'=\theta$：**

$D_{KL}(\pi_{\theta}||\pi_{\theta'})=\underbrace{\sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta}}(a|s)}_{与\theta'无关，常数}- \sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta'}}(a|s)=-\sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta'}}(a|s)$

$\nabla_{\theta'}D_{KL}(\theta)=\nabla_{\theta'} D_{KL}(\theta)^\top \Delta\theta=-\sum \pi_{\boldsymbol{\theta}}(a|s) \nabla_{\theta'} \log \pi_{\boldsymbol{\theta'}}(a|s)=-\sum \underbrace{\pi_{\boldsymbol{\theta}}(a|s) \cdot \dfrac{1}{\pi_{\boldsymbol{\theta'}}(a|s)}}_{\theta'=\theta,这里等于1} \cdot \nabla_{\theta'} \pi_{\boldsymbol{\theta'}}(a|s)=-\sum \nabla_{\theta'} \pi_{\boldsymbol{\theta'}}(a|s)$

因为 策略是合法概率分布 $\sum \pi_{\theta}(a|s) = 1$ ，$\nabla_{\theta} \sum \pi_{\theta}(a|s)=0$

$\nabla_{\theta'}D_{KL}(\theta)=0$

**最终：**

$D_{KL}(\pi_{\theta}||\pi_{\theta'}) \approx \dfrac{1}{2}\Delta\theta^\top H \Delta\theta$




