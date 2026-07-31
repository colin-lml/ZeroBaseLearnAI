

# 海森矩阵（**黑塞矩阵**Hessian matrix）

对多元标量函数：$ f(\boldsymbol{x}) = f(x_1,x_2,\dots,x_n)$

**梯度**（一阶偏导数$\partial$向量  ) 

$\boxed{\nabla f(\boldsymbol{x})=\begin{bmatrix}\frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\   \frac{\partial f}{\partial x_n} \end{bmatrix}}$



**海森矩阵$H$**： 梯度再求导

$\boxed{\nabla{^2} f(\boldsymbol{x})=\begin{bmatrix}\
\frac{\partial^{2} f}{\partial x_1^2} & \frac{\partial^{2} f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^{2} f}{\partial x_1 \partial x_n}\ 
\\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \dots & \dots &  \frac{\partial^{2} f}{\partial x_2 \partial x_n} \
\\ \vdots  & \vdots & \vdots & \vdots \
\\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \dots & \dots & \frac{\partial^2 f}{\partial x_n^2 } \
\end{bmatrix}}$



**重要性质**

如果函数二阶连续可微：**混合偏导可交换**

$\boxed{\frac{\partial^2 f}{\partial x_i \partial x_j}=\frac{\partial^2 f}{\partial x_j \partial x_i}}$

**海森矩阵一定对称**



**二次型**

 二次型通用公式 : $\boxed{f(\boldsymbol{x}) = \boldsymbol{x}^\top A \boldsymbol{x}}$ 

- $\boldsymbol{x}$是 $[n\times1]$ 向量

- $A$ 是 $[n\times n]$ 方阵

- 求梯度推导：$ \nabla f(\boldsymbol x)= \nabla(\boldsymbol{x}^\top A \boldsymbol{x})=(A+A^{\top})\boldsymbol{x}$

- 再次求导：$\nabla^2 f(\boldsymbol x)=\nabla (A+A^{\top})\boldsymbol{x}=A+A^{\top}$

若 $\boldsymbol{A}$ 对称($A==A^{\top}$)：

- [ ] $\quad \nabla f(\boldsymbol x)=\nabla(\boldsymbol{x}^\top A \boldsymbol{x})=2A\boldsymbol x$

- [ ] $\quad \nabla^2 f(\boldsymbol x)=2A=H$  **海森矩阵$H$**

通过二次型公式求出$A$方阵，用矩阵运算代替求导过程

**代替求导：** $\nabla f(\boldsymbol x)=\nabla(\boldsymbol{x}^\top A \boldsymbol{x})=2A\boldsymbol x$

**海森矩阵：** $\nabla^2 f(\boldsymbol x)=2A=H$



**二维实例**

方程式：$f(x_1,x_2)=x_1^2 + 2x_1x_2 + 3x_2^2$

二次型通用形式

$\boldsymbol{x}=\begin{bmatrix}x_1\\x_2\end{bmatrix} ,\quad f(\boldsymbol{x}) = \boldsymbol{x}^\top A \boldsymbol{x}$

展开：

$f(\boldsymbol{x})=\underbrace{\begin{bmatrix}x_1&x_2\end{bmatrix}}_{\boldsymbol{x}^\top} \quad  
\underbrace{\begin{bmatrix}a_{11} & a_{12}\\ a_{21} & a_{22} \end{bmatrix}}_{A} \quad
\underbrace{\begin{bmatrix}x_1\\x_2\end{bmatrix}}_{\boldsymbol{x}} = 
\underbrace{a_{11}x_1^2+(a_{12}+a_{21})x_1x_2+a_{22}x_2^2}_{方程式}$

要求矩阵**对称** 所以：

$a_{12}=a_{21}， A= \begin{bmatrix} 1 & 1 \\ 1 &3 \end{bmatrix}$

知道$A$矩阵带入求导公式：

 $ \nabla f(\boldsymbol x)=2A\boldsymbol x= 2\begin{bmatrix} 1 & 1 \\ 1 &3 \end{bmatrix} \begin{bmatrix}x_1\\x_2\end{bmatrix}=\begin{bmatrix}2x_1+2x_2\\2x_1+6x_2\end{bmatrix}$

  $ \nabla^2 f(\boldsymbol x)=2A=H=\begin{bmatrix} 2 & 2 \\ 2 & 6 \end{bmatrix}$



# 泰勒公式

近似求解：在一个光滑函数上，选定某一点，构造多项式（方程式），在该点局部近似代替原本复杂的函数，用泰勒展开式作近似求解， 想要逼近效果更好，就要捕捉更多特征

- **0 阶：** 只复制该点函数值，只有一个点，完全没有变化信息。

- **1 阶：** 加上**一阶导数（斜率）** → 切线近似（直线）

- **2 阶：** 再加**二阶导数（曲率）** → 抛物线，可以描述曲线怎么弯 （一般只作二阶）
  
  

设函数 $f(x)$ 在 $x_0$ 光滑，$\Delta x = x-x_0$



**带皮亚诺余项** ：

 $\boxed{f(\Delta x +x_0)= \underbrace{\frac{f(x_0)}{!0}  }_{0阶}+ \underbrace{\frac{f'(x_0)\Delta x}{1!}}_{1阶} + \underbrace{\frac{f''(x_0)(\Delta x)^2 }{2!}}_{2阶}+  \underbrace{\frac{f'''(x_0)(\Delta x)^3 }{3!}}_{3阶} + \dots} $



**二阶截断：**
一元函数：
$\boxed{f(\Delta x +x_0) \approx \underbrace{f(x_0)}_{常数基准} + \underbrace{f'(x_0)\Delta x}_{切线、斜率信息} + \underbrace{\frac{1}{2} f''(x_0) (\Delta x)^2}_{曲率信息}}$

多元函数：

$\boxed{f(\Delta x +x_0) \approx f(x_0) + \nabla f(x_0)^{\top}(\Delta x) +\frac{1}{2}(\Delta x)^{\top} \nabla^2 f(x_0) \Delta x}$ 

结合二次型公式用矩阵$A$代替求导

1. $\boxed{\nabla f(x_0){^\top} = (2Ax_0)^{\top}=>2x_0^{\top}A}$

2. $\boxed{\nabla^2 f(x_0)=2A=H}$

$\boxed{f(\Delta x +x_0) \approx \underbrace{f(x_0)}_{x_0^{\top}Ax_0} + \underbrace{\nabla f(x_0)^{\top} \Delta x}_{2x_0^{\top}A(\Delta x)} +(\Delta x)^{\top} A \Delta x}$



**数据演算**

方程式：$f(x_1,x_2)=x_1^2 + 2x_1x_2 + 3x_2^2$

$\textcircled{1}已知值 \begin{cases} A= \begin{bmatrix} 1 & 1 \\ 1 &3 \end{bmatrix}\\[4pt]
\\ 中心点: x_0=  \begin{bmatrix} 2  \\ 1  \end{bmatrix}\\[4pt]
\\增量: \Delta x=  \begin{bmatrix} 1  \\ 2  \end{bmatrix} \\[4pt]
\\目标: \ x= x_0+ \Delta x=\begin{bmatrix} 3  \\ 3  \end{bmatrix} 
\end{cases}$



$\textcircled{2}逐项计算 \begin{cases} f(x_0)=f(2,1)= \underbrace{\begin{bmatrix} 2 & 1 \end{bmatrix}}_{x_0^{\top}} \underbrace{\begin{bmatrix} 1 & 1 \\ 1 &3 \end{bmatrix}}_{A} \underbrace{\begin{bmatrix} 2 \\ 1 \end{bmatrix}}_{x_0}=11\\[1pt]
\\ 2x_0^{\top}A(\Delta x)= 2 * \underbrace{\begin{bmatrix} 2 & 1 \end{bmatrix}}_{x_0^{\top}} \underbrace{\begin{bmatrix} 1 & 1 \\ 1 &3 \end{bmatrix}}_{A} \underbrace{\begin{bmatrix} 1 \\ 2 \end{bmatrix}}_{\Delta x}=26\\[4pt]
\\(\Delta x)^{\top} A \Delta x= \underbrace{\begin{bmatrix} 1 & 2 \end{bmatrix}}_{(\Delta x)^{\top}} \underbrace{\begin{bmatrix} 1 & 1 \\ 1 &3 \end{bmatrix}}_{A} \underbrace{\begin{bmatrix} 1 \\ 2 \end{bmatrix}}_{\Delta x}=17 \\[4pt]
\\f(\Delta x +x_0) \approx 11+26+17=54 \end{cases}$



$\textcircled{3}核验真值\begin{cases}二阶泰勒: f(\Delta x +x_0) \approx 11+26+17=54  \\
f(3,3)=3^2 + 2*3*3 + 3*3^2=9+18+27=54
\end{cases}$



# 小结

- **海森矩阵$H$** 通过**二次型公式** $\boxed{f(\boldsymbol{x}) = \boldsymbol{x}^\top A \boldsymbol{x}}$ 构造出$A$矩阵，借助矩阵运算代替求导运算，若 $A$ 对称，梯度再次求导等于得到海森矩阵$2A=H$

- **泰勒展开式** 代替原本函数作**近似求解** ，借助于海森矩阵

- **二次型公式** 为了求导后消去系数 $2$ 使得$A=H$, 所以加上$\dfrac{1}{2}$ 公式: $\boxed{f(\boldsymbol{x}) = \dfrac{1}{2}\boldsymbol{x}^\top H \boldsymbol{x},A=H }$

- **二阶泰勒：**${f(\Delta x +x_0) \approx \underbrace{f(x_0)}_{\frac{1}{2} x_0^{\top}Hx_0} + \underbrace{x_0^{\top}H\Delta x}_{\nabla f(x_0)^{\top} \Delta x} +\frac{1}{2}(\Delta x)^{\top} H \Delta x}$

- **化简写法：**  二次型函数 ${f(\boldsymbol{x}) = \frac{1}{2} \boldsymbol{x}^\top A \boldsymbol{x}}$ ，令$\boldsymbol{d} = \Delta \boldsymbol{x}; $ 则 $\boxed{{f( \boldsymbol x_0 + \boldsymbol d) \approx} f(x_0) + \nabla f(x_0)^{\top} d + \frac{1}{2}d^{\top}Hd}$

$\boxed{{f( \boldsymbol x_0 + \boldsymbol d) \approx} f(x_0) + \nabla f(x_0)^{\top} d + \frac{1}{2}d^{\top}Hd}$ 

- 若$A=H$对称且${f(\boldsymbol{x}) = \frac{1}{2} \boldsymbol{x}^\top A \boldsymbol{x}}$ 则
  $\boxed{f( \boldsymbol x_0 + \boldsymbol d) \approx f(x_0) + x_0^{\top}H d + \frac{1}{2}d^{\top}Hd}$
  
  
  
  

# 牛顿法迭代

**目标：** 求解无约束极小  $\min {f(\boldsymbol{x}) = \frac{1}{2} \boldsymbol{x}^\top A \boldsymbol{x}}$， $A=H$对称

**二阶泰勒：** $ f( \boldsymbol x_0 + \boldsymbol d) \approx f(x_0) + x_0^{\top}H d + \frac{1}{2}d^{\top}Hd$

**牛顿法核心思想** 当前位置 $\boldsymbol x_k$，**局部用二阶泰勒近似代替原函数**

构造局部近似函数：$\phi(d) = f(x_k)+ (x_k)^{\top} H d + \frac12 d^{\top} H d$

- $\phi$ 函数记号，指二阶泰勒近似函数

- $\phi(d)$ ：从 $x_k$ 偏移 ${d}$ 后 使得$f(\boldsymbol x)$出现极小值 ，找出最优$\boldsymbol{d}^*$ 就是牛顿方向 $\boldsymbol{d}_k$

- 对 $\phi(\boldsymbol{d})$ 求导后令导数为0，得到极小值



$\phi(d) = \underbrace{f(x_k)}_{常量}+ \underbrace{(x_k)^{\top} H d}_{\nabla 项一} + \underbrace{\frac12 d^{\top} H d}_{\nabla 项二}$

**求导规则：**

- $\nabla_{\boldsymbol x} (\boldsymbol a^{\top} \boldsymbol x) = \boldsymbol a$ 其中$a$是向量

- $\nabla_{\boldsymbol x} (\boldsymbol x^{\top}\boldsymbol a) = \boldsymbol a$ 其中$a$是向量

- x$\nabla_{\boldsymbol x}(\boldsymbol x^{\top} A \boldsymbol x) = \boldsymbol{A x} + A^{\top} \boldsymbol x$ 其中$x$是向量，$A$是矩阵
  
  

$\nabla$ 项一 ：$(x_k)^{\top}H=a^{\top}$是常数行向量  所以 $\nabla_d (a^{\top} d)=a^{\top}=((x_k)^{\top}H)^{\top}=Hx_k$

$\nabla$ 项二: $\nabla_d(\frac12 d^{\top} H d)=Hd$



对 $\phi(\boldsymbol{d})$ 求导结果：$\nabla_d \phi(d)=0 +Hx_k+Hd$ ，求极小值$\nabla \phi(\boldsymbol{d})=0$置零$Hx_k+Hd=0$

$Hd=-Hx_k$ ，左右同时乘以 $H^{-1}$

$d=-x_k$



**如何求$H^{-1}$矩阵**

$H=\begin{bmatrix}a& b\\ c&d\end{bmatrix} \Rightarrow H^{-1}=\dfrac{1}{ad-bc} \begin{bmatrix}d & -b\\ -c&a\end{bmatrix}$

行列式:$det(H)=ad-bc$

记忆口诀：主对角线互换 $a\leftrightarrow d$，副对角线变号 $-b,-c$，整体除以行列式。



**关键结论**

1. 任意正定二次函数 $\displaystyle f=\frac12\boldsymbol x^\top H\boldsymbol x$

2. 牛顿方向 $\boldsymbol d_k=-\boldsymbol x_k$

3. 步长$\alpha=1$，**单次迭代直达全局最优**



# 最速下降法

- **用一阶泰勒**   $ f( \boldsymbol x_0 + \boldsymbol d) = f(x_0) + x_0^{\top}H d $

- **核心思想** 局部下降最快的方向 = 负梯度方向

- 构造局部近似函数 ：$\phi(d) = f(x_k)+ (x_k)^{\top} H d$ ，其中$f(d)=(x_k)^{\top}Hd$ 是线性函数 ,线性近似不存在最小点

- 我们只能限定：**只允许单位长度方向**，只优化「方向」，不能无限走远。

- 增加约束条件$\|\boldsymbol{d}\|=1$则 $\phi(d) = f(x_k)+ (x_k)^{\top} H d=f(x_k)+\underbrace{G_k^{\top}}_{\nabla f(x_k)} d $其中$f(x_k)$是常量

- $\min_{||d||=1} \phi(d)=G_k^{\top}d$

✅结论：**单位向量下，使线性近似函数$\phi(\boldsymbol{d})$最小的方向 = 归一化负梯度方向。**我们得到**最优下降方向**

✅结论: $\boldsymbol{d}_k = -\boldsymbol{G}_k = -\nabla f(\boldsymbol{x}_k)$


