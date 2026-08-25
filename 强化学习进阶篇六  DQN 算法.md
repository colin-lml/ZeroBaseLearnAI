# DQN

## DQN 由来

 在Q-learning 算法中用一张存储每个状态下所有动作$Q$值的表格，但在车杆环境中它的状态值就是连续的（无限个状态），  动作值是离散的(向左施加力或向右施加力)，因此不能用$Q$表，使用**函数拟合**来求解，用一神经网络来表示函数$Q$

神经网络的输入是状态$S$，输出是动作$a$对应的价值 （预期累计折扣奖励），这是DQN（Deep Q-Network，深度 Q 网络）的由来。



## DQN公式推导

回顾一下 Q-learning 的更新规则： $Q(s,a) \leftarrow Q(s,a)+ \alpha[r+ \gamma \max \limits_{a} Q(s',a')-Q(s,a)]$ ,我们用神经网络代替了$Q$表，记$Q_{dqn}$表示

1. 公式： $Q_{\theta}(s,a) \leftarrow Q_{\theta}(s,a)+ \alpha[r+ \gamma \max \limits_{a} Q_{\theta}(s',a')-Q_{\theta}(s,a)]$

2. 因为$Q_{dqn}$是神经网络所以要更新只能更新神经网络权重，也就是训练神经网络。

3. 中括号部分叫作 TD Error（时序差分误差）：$\delta=\underbrace{r+ \gamma \max \limits_{a} Q_{\theta}(s',a')}_{目标值(标签)}-\underbrace{Q_{\theta}(s,a)}_{当前预测值}$ ，使用两套 一样 网络，**在线网络** 预测当前值，实时更新。**目标网络** 表示目标值，定期更新。可以这样理解在Q-learning 中更新迭代$Q$表时 先更新当前的$Q(a,s)$，再更新下个$Q(a',s')$ 

4. 定义损失函数 $\mathcal L_{\text{MSE}} = \big(目标网络 - 在线网络\big)^2$

**总结：**  使用神经网络$Q_{\theta}$ 代替$Q$表，损失函数用均方误差$MSE$ $\mathcal L_{\text{MSE}} = \big(目标网络 - 在线网络\big)^2$  训练神经网络。











---

2. Bellman 最优方程
   执行动作 $a$ 后，环境从 $s$ 转移到下一状态 $s'$，获得即时奖励 $r$。
   后续最优策略会从 $s'$ 中选择价值最大的动作，因此：
   $
   Q^*(s,a)
   \mathbb{E} \left[ r+\gamma\max_{a'}Q^*(s',a') \right] $
   这就是 Bellman 最优方程。
   若当前回合已经结束，则没有未来奖励：
   $$Q^*(s,a)=r $
   可以合并为：
   $
   Q^*(s,a)
   r+ \gamma(1-done) \max_{a'}Q^*(s',a') $
   其中：
   •    done = 1：回合结束，未来价值部分为 0；
   •    done = 0：回合未结束，保留未来价值。

---

3. 表格型 Q-Learning 更新
   如果使用表格直接保存每个 $Q(s,a)$，更新公式是：
   $ Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r+\gamma(1-done)\max_{a'}Q(s',a') -Q(s,a) \right]$ 
   中括号部分叫作 TD Error（时序差分误差）：
   $
   \delta=
   \underbrace{
   r+\gamma(1-done)\max_{a'}Q(s',a')
   }_{\text{c target}}-
   \underbrace{Q(s,a)}_{\text{当前预测值}} $
   含义是：
   用“实际得到的奖励 + 对未来的估计”，修正当前 Q 值预测。

---

4. DQN：使用神经网络表示 Q 函数
   DQN 不再维护一张 Q 表，而是使用参数为 $\theta$ 的神经网络：
   $ Q_\theta(s,a) $
   在 CartPole 中，网络输入状态 $s$，输出所有动作的 Q 值：
   $
   Q_\theta(s)
   [ Q_\theta(s,a_0), Q_\theta(s,a_1) ] $
   例如：
   $Q_\theta(s)=[2.3, 3.8] $
   表示：
   •    向左的预期累计奖励是 2.3；
   •    向右的预期累计奖励是 3.8。
   若实际执行动作是 $a=1$，则取：
   $ Q_\theta(s,1)=3.8 $
   
   

5. DQN 的目标值
   DQN 引入目标网络 $Q_{\theta^-}$，用于计算相对稳定的目标值：
   $ y= r+\gamma(1-done) \max_{a'}Q_{\theta^-}(s',a')$ 
   这里：
   •    $Q_{\theta}$：在线网络，正在被训练；
   •    $Q_{\theta^-}$：目标网络，定期从在线网络复制参数；
   •    $y$：监督学习中的“标签”或“目标值”。

6.损失函数推导

神经网络预测的是：
$Q_\theta(s,a) $
Bellman 方程给出的训练目标是：
$ y= r+\gamma(1-done) \max_{a'}Q_{\theta^-}(s',a') $
两者之间使用均方误差：
$
L(\theta)
\frac{1}{N} \sum_{i=1}^{N} \left( y_i-Q_\theta(s_i,a_i) \right)^2 $
其中 $N$ 是批量样本数量。
代入完整形式：
$
L(\theta)
\frac{1}{N}
\sum_{i=1}^{N}
\left[
r_i+
\gamma(1-done_i)
\max_{a'}Q_{\theta^-}(s'_i,a')
Q_\theta(s_i,a_i) \right]^2 $
然后对网络参数 $\theta$ 求梯度：
$
\theta
\leftarrow
\theta
\eta \nabla_\theta L(\theta) $
其中 $\eta$ 是学习率。
代码中由 libtorch 完成反向传播和优化：



DQN 的核心可写为：
$ \boxed{ y= r+\gamma(1-done) \max_{a'}Q_{\theta^-}(s',a') } $
$ \boxed{ L(\theta)= \left[ y-Q_\theta(s,a) \right]^2 } $
$ \boxed{ \theta \leftarrow \theta-\eta\nabla_\theta L(\theta) }$
简单理解：

1. 网络预测当前动作值 $Q_\theta(s,a)$；
2. 使用奖励和下一状态估计目标值 $y$；
3. 计算预测值与目标值的差距；
4. 通过反向传播调整网络参数；
5. 让网络以后对类似状态给出更准确的 Q 值。
   
   
   
   
   
   

$
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
\left[
r
+
\gamma\max_{a'}Q(s',a')
Q(s,a) \right]$ 
