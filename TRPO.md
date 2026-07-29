

基础定义

对多元标量函数：$ f(\boldsymbol{x}) = f(x_1,x_2,\dots,x_n)$

**梯度**（一阶偏导数$\partial$向量  ) 

$\nabla f=\begin{bmatrix}\frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\   \frac{\partial f}{\partial x_n} \end{bmatrix}$



**海森矩阵：** 梯度再求导

$\nabla{^2} f=\begin{bmatrix}\
\frac{\partial^{2} f}{\partial x_1^2} & \frac{\partial^{2} f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^{2} f}{\partial x_1 \partial x_n}\ 
\\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \dots & \dots &  \frac{\partial^{2} f}{\partial x_2 \partial x_n} \
\\ \vdots  & \vdots & \vdots & \vdots \
\\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \dots & \dots & \frac{\partial^2 f}{\partial x_n^2 } \
\end{bmatrix}$



**重要性质**

如果函数二阶连续可微：**混合偏导可交换**

$\frac{\partial^2 f}{\partial x_i \partial x_j}=\frac{\partial^2 f}{\partial x_j \partial x_i}$

**海森矩阵一定对称**



**引入矩阵 $\boldsymbol{A}$**







**二维实例**

方程式：$f(x_1,x_2)=x_1^2 + 2x_1x_2 + 3x_2^2$

二次型通用形式

$\boldsymbol{x}=\begin{bmatrix}x_1\\x_2\end{bmatrix} ,\quad f(\boldsymbol{x}) = \boldsymbol{x}^\top A \boldsymbol{x}$

展开：

$f(\boldsymbol{x})=\begin{bmatrix}x_1&x_2\end{bmatrix}\
\begin{bmatrix}a_{11} & a_{12}\\ a_{21} & a_{22} \end{bmatrix}\
\begin{bmatrix}x_1\\x_2\end{bmatrix} = a_{11}x_1^2+(a_{12}+a_{21})x_1x_2+a_{22}x_2^2$

要求矩阵**对称** 所以$a_{12}=a_{21}$

$A= \begin{bmatrix} 1 & 1 \\ 1 &3  \end{bmatrix}$




