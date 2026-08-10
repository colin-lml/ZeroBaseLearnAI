## 1.原始定义

$\underbrace{\pi_{\boldsymbol{\theta}}(a|s)}_{动作概率分布}= \underbrace{\mathrm{softmax}\Big(w_2\, \max\big(w_1 s + b_1,\,0\big)+b_2\Big)}_{策略神经网络},\quad \boldsymbol{\theta}= \underbrace {\{w_1,b_1,w_2,b_2\}}_{网络参数}$



$\boldsymbol{L(\theta')=\mathbb{E}_{s\sim\rho_\theta,\,a\sim\pi_\theta}\left[\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)} \cdot A_\theta(s,a)\right]}$

符号说明

* $\pi_\theta$：**旧策略，固定不动**
* $\pi_{\theta'}$：新策略，变量，我们优化 $\theta'$
* $A_\theta(s,a)$：**优势函数 Advantage**
* $\dfrac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}$：重要性采样比率（probability ratio，PPO 里也用）
* 期望是用**旧策略采样出来的数据**，数据不重新采样

**最大化替代目标，同时约束新($\pi_{\theta'}$)旧($\pi_\theta$)策略 KL 散度不超过阈值**，避免策略一次性更新过大导致训练崩塌。





## 2.泰勒展开

令 $\theta'=\theta+\Delta\theta$，$\Delta\theta$ 是参数增量。

一阶泰勒：

$L(\theta+\Delta\theta)\approx \underbrace{L(\theta)}_{常量丢掉}+\underbrace{\nabla_{\theta}L(\theta)^\top}_{g=\nabla_{\theta}L(\theta)} \Delta\theta$



$L$**一阶展开** 优化简化为: $\max g^\top \Delta\theta$



## 3. $D_{KL}$ 散度

令 $\theta'=\theta+\Delta\theta$，$\Delta\theta$ 是参数增量，$\pi_{\theta}=\pi_{\theta}(a|s)$



$D_{KL}(\pi_{\theta}||\pi_{\theta'})=\sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta}}(a|s)- \sum \pi_{\boldsymbol{\theta}}(a|s)\log \pi_{\boldsymbol{\theta'}}(a|s)$



## 4.$D_{KL}$二阶泰勒展开：

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



## 5.TRPO目标约束

$\begin{cases} 
\max g^\top \Delta\theta \\
s.t.\quad D_{KL}(\pi_{\theta}||\pi_{\theta'}) \approx \dfrac{1}{2}\Delta\theta^\top H \Delta\theta \le \delta 
\end{cases}$

$\boldsymbol{\delta}：$ **KL 散度最大允许上界（超参数）**，人为设定的常数。



## 6.拉格朗日乘子法(KKT)

$\begin{cases} \max g^\top \Delta\theta \\s.t.\quad  \dfrac{1}{2}\Delta\theta^\top H \Delta\theta \le \delta \end{cases}$



  令 $x=\Delta\theta$

**不等式约束:**

 $h(\boldsymbol x)=\dfrac12\boldsymbol x^\top H\boldsymbol x-\delta \le 0$

$\begin{cases} \max g^\top \boldsymbol x == {\color{red} {-}} \min g^\top \boldsymbol x \\s.t.\quad h(\boldsymbol x)=\dfrac12\boldsymbol x^\top H\boldsymbol x-\delta \le 0 \end{cases}$



**构造拉格朗日函数**

$\mathcal L(\boldsymbol x,\lambda)= \boldsymbol g^\top \boldsymbol x {\color{red}{-}} \lambda\left(\frac12\boldsymbol x^\top H\boldsymbol x-\delta\right)$

$\lambda\ge 0$ 是拉格朗日乘子。



### KKT 条件

$\begin{cases}
 驻点条件: \nabla_{x}\mathcal L =  \nabla_{x} g^\top x- \nabla_{x}\lambda\left(\frac12 x^\top H x-\delta\right)=g- \lambda Hx=0\\
 原可行性: \frac12 x^\top H x \le \delta\\
 对偶可行性: \lambda \ge 0\\
 互补松弛: \lambda\left(\frac12 x^\top H x-\delta\right)=0
\end{cases}$

$\textcircled{1}求解\lambda= \begin{cases}g- \lambda Hx=0 \quad \Rightarrow\ x=\frac{1}{\lambda}  H^{-1} g \quad 待解出\lambda后算出x\\
\frac12 x^\top H x-\delta=0 \quad \Rightarrow  x^\top Hx=2\delta \Rightarrow\ \underbrace{(\frac{1}{\lambda} H^{-1}g)^\top H (\frac{1}{\lambda} H^{-1}g)=2\delta}_{可以解出\lambda}\\
 解过程1：\dfrac{1}{\lambda^2} g^\top H^{-1} \underbrace {H H^{-1}}_{HH^{-1}=I}g=2\delta\\\\
 解过程2: g^\top H^{-1}g =2\delta  \lambda^2 \Rightarrow\ \dfrac{g^\top H^{-1}g}{2\delta}=\lambda^2\\\\
 \lambda=\sqrt{\dfrac{g^\top H^{-1}g}{2\delta}}
\end{cases}$

$\textcircled{2}求解 x= \begin{cases}\\  \lambda=\sqrt{\dfrac{g^\top H^{-1}g}{2\delta}}\\\\
x=\frac{1}{\lambda} H^{-1}g = \dfrac{H^{-1}g}{\sqrt{\dfrac{g^\top H^{-1}g}{2\delta}}}\quad 根号分式：\sqrt{\dfrac{A}{B}}=\dfrac{\sqrt A}{\sqrt B}\\\\
x=\sqrt{\dfrac{2\delta}{g^\top H^{-1}g}} \cdot H^{-1}g
\end{cases}$



$\textcircled{3}=\begin{cases}\\  x=\Delta\theta=\sqrt{\dfrac{2\delta}{g^\top H^{-1}g}} \cdot H^{-1}g \\\\  \end{cases}$



## 7.线性搜索(回溯线搜索)








