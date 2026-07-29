

# 海森矩阵（**黑塞矩阵**（Hessian matrix））

对多元标量函数：$ f(\boldsymbol{x}) = f(x_1,x_2,\dots,x_n)$

**梯度**（一阶偏导数$\partial$向量  ) 

$\nabla f(\boldsymbol{x})=\begin{bmatrix}\frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\   \frac{\partial f}{\partial x_n} \end{bmatrix}$



**海森矩阵$H$**： 梯度再求导

$\nabla{^2} f(\boldsymbol{x})=\begin{bmatrix}\
\frac{\partial^{2} f}{\partial x_1^2} & \frac{\partial^{2} f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^{2} f}{\partial x_1 \partial x_n}\ 
\\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \dots & \dots &  \frac{\partial^{2} f}{\partial x_2 \partial x_n} \
\\ \vdots  & \vdots & \vdots & \vdots \
\\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \dots & \dots & \frac{\partial^2 f}{\partial x_n^2 } \
\end{bmatrix}$



**重要性质**

如果函数二阶连续可微：**混合偏导可交换**

$\frac{\partial^2 f}{\partial x_i \partial x_j}=\frac{\partial^2 f}{\partial x_j \partial x_i}$

**海森矩阵一定对称**



**二次型**

 二次型通用公式 : $f(\boldsymbol{x}) = \boldsymbol{x}^\top A \boldsymbol{x}$ 

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

 $ \nabla f(\boldsymbol x)=2A\boldsymbol x= 2\begin{bmatrix} 1 & 1 \\ 1 &3 \end{bmatrix} \begin{bmatrix}x_1\\x_2\end{bmatrix}=\begin{bmatrix}2x_1+2x_2\\x_1+3x_2\end{bmatrix}$

  $ \nabla^2 f(\boldsymbol x)=2A=H=\begin{bmatrix} 2 & 2 \\ 2 & 6 \end{bmatrix}$



# 泰勒公式

近似求解：在一个光滑函数上，选定某一点，构造多项式（方程式），在该点局部近似代替原本复杂的函数，用泰勒展开式作近似求解， 想要逼近效果更好，就要捕捉更多特征

- **0 阶：** 只复制该点函数值，只有一个点，完全没有变化信息。

- **1 阶：** 加上**一阶导数（斜率）** → 切线近似（直线）

- **2 阶：** 再加**二阶导数（曲率）** → 抛物线，可以描述曲线怎么弯 （一般只作二阶）
  
  

设函数 $f(x)$ 在 $x_0$ 光滑，$\Delta x = x-x_0$



**带皮亚诺余项** ：

 $f(\Delta x +x_0)= \underbrace{\frac{f(x_0)}{!0}  }_{0阶}+ \underbrace{\frac{f'(x_0)\Delta x}{1!}}_{1阶} + \underbrace{\frac{f''(x_0)(\Delta x)^2 }{2!}}_{2阶}+  \underbrace{\frac{f'''(x_0)(\Delta x)^3 }{3!}}_{3阶} + \dots $



**二阶截断：**
一元函数：
$f(\Delta x +x_0) \approx \underbrace{f(x_0)}_{常数基准} + \underbrace{f'(x_0)\Delta x}_{切线、斜率信息} + \underbrace{\frac{1}{2} f''(x_0) (\Delta x)^2}_{曲率信息}$

多元函数：

$f(\Delta x +x_0) \approx f(x_0) + \nabla f(x_0)^{\top}(\Delta x) +\frac{1}{2}(\Delta x)^{\top} \nabla^2 f(x_0) \Delta x$ 

结合二次型公式用矩阵$A$代替求导

1. $\nabla f(x_0){^\top} = (2Ax_0)^{\top}=>2x_0^{\top}A$，

2. $\nabla^2 f(x_0)=2A$

$f(\Delta x +x_0) \approx f(x_0) + 2x_0^{\top}A(\Delta x) +(\Delta x)^{\top} A \Delta x$



**赋值演算**

方程式：$f(x_1,x_2)=x_1^2 + 2x_1x_2 + 3x_2^2$

$\textcircled{1} \textcircled{1}\;$


