# 导数

导数是微积分中的核心概念之一，函数在某一点的导数就是该函数所代表的曲线在这一点上的切线斜率（不清楚的话，建议找相应教程）， 常用的导数公式有

**常数函数**

若 f(x) = C（C 为常数），则f'(x) = 0

**幂函数**

$若 (f(x) = x^n)（n 为常数），则f'(x) = n *x^{n-1}$

示例 $f(x)=x^2, 则f'(x)=2x^1$

**指数函数(暂时用不上)**

$自然指数函数：若 f(x) = e^x，则f'(x) = e^x$

$一般指数函数：若 f(x) = a^x，则f'(x)= a^x * ln(a)$



**三角函数（暂时用不上）**

* f(x) = sin (x)，则 f'(x) = cos (x)
* f(x) = cos (x)，则 f'(x) = -sin (x)

****反三角函数（暂时用不上）****



**导数的运算法则**

$设函数 f(x) 和 g(x) 均可导$

1.****和差法则****

$[f(x)+g(x)]'$ = $f'(x)+g'(x)$

**乘积法则**

$[f(x)*g(x)]'$=$f'(x)*g(x)+f(x)*g'(x)$

_特别地，若 g(x)=C 为常数则: $[f(x)*g(x)]'$=C * f'(x)+f(x)*0_

**复合函数求导法则（链式法则）**

$[f(g(x))]', 令 t=g(x)$ ；$[f(g(x))]'=f'(t)*g'(x)$

示例： $y=sin(x^2)$ 则：$cos(x^2) * 2*x^1$ 

以上是在神经网络计算过程中要使用的数学基础    

1.$u=w5*x$                                      <--- ($w5*outh1+b2=nety1=u$)

2.$z=(sigmoid(u)-o1)$              <--- $outy1=sigmoid(nety1)$ 

3.$E=\frac{1}{2}z^2$                                        <----($E=\frac{1}{2}(outy1-o1)^2$) ,$z=(outy1-o1)$

4.$E'(w5)=E'*z'*u'$

5.$E'=z=>y-o1=>(outy1-o1)$

6.$z'=(sigmoid(u)'+0)=>sigmoid(u)'=>sigmoid_derivative(outy1)$

7.$u'=x*(w5^0)=x=outh1$

8.$E'(w5)=(outy1-o1)*sigmoid_derivative(outy1)*outh1$

$E=\frac{1}{2}(outy1-o1)^2$



1. $neth1=w1*i1+b1$  记作 $U=w1*i1+b1$

2. $outh1=sigmoid(neth1)$ 记作 $K=sigmoid(neth1)$

3. $nety1=outh1*w5+b2$ 记作 $G=outh1*w5+b2$

4. $nety2=outh1*w7+b2$ 记作 $F=outh1*w7+b2$

5. $Z=sigmoid(nety1)-o1 ,  outy1=sigmoid(nety1)$

6. $T=sigmoid(nety2)−o2, outy2=sigmoid(nety2)$

7. 损失函数$Ez(x)=\frac{1}{2}x^2，Et(x)=\frac{1}{2}x^2$。总损失复合函数$E=Ez+Et$

8. $E$ 对w1的偏导数为$E(w1)'=Ez(w1)'+Et(w1)'$

9. $Ez(w1)'=Ez'*Z'*G'*K'*U'$

10. $Et(w1)'=Et'*T'*F'*K'*U'$

11. $E(w1)'=Ez'*Z'*G'*K'*U'+Et'*T'*F'*K'*U'=(Ez'*Z'*G'+Et'*T'*F')*K'∗U'$

12. $Ez'=(outy1-o1),Et'=(outy2-o2),U'=1*i1+0$

13. $Z'=sigmoid_derivative(outy1)-0,T'=sigmoid_derivative(outy1)-0,$ 

14. $F(outh1)'=(1*outh1^0)*w7+0,G(outh1)'=(1*outh1^0)*w5+0,$

15. $K'=sigmoid_derivative(outh1)$

$$
E(w1)′=((outy1−o1)*sigmoid_derivative(outy1)*w5+(outy2−o2)*sigmoid_derivative(outy2)*w7)*sigmoid_derivative(outh1)*i1
$$



$T=w5*h1$+0

$F=sigmoid(T(w5)),展开式为 F(w5)=sigmoid(w5*h1)$

$E(o1,o2)=E(o1)+0，那么E=E(o1)这里o1不是真实值而是变量F(w5)它展开式为$

$E=\frac{1}{2}(sigmoid(w5*h1)-o1)^2$要对$w5$求导,用复合函数求导法则（链式法则）,$(sigmoid(w5*h1)-o1)$当成函数$Z=(sigmoid(w5*h1)-o1)$那么E对$w5$的导数$E(w5)=2*\frac{1}{2}(sigmoid(w5*h1)-o1)*Z'$



1.$E(w5)'=(sigmoid(w5*h1)-o1)*Z'$ ,  $sigmoid(w5*h1)-o1$ 我们是知道值的，代码中Loss_derivative(outy1,o1)函数计算,简写(outy1-o1)

$E(w5)'=(outy1-o1)*Z'$

2.对$Z$求导用和差法则 $Z(w5)'=[sigmoid(w5*h1)]'-[o1]'=[sigmoid(w5*h1)]', o1$是常量导数为0，$F(w5)=sigmoid(w5*h1)$所以

$Z(w5)'=F(w5)'$

3.$E(w5)'=(outy1-o1)*F(w5)'$, $F$又嵌套一个$T=w5*h1$函数，所以$F(w5)'=sigmoid(w5*h1)'*T(w5)'$, $sigmoid(w5*h1)$的导数也是知道值的用sigmoid_derivative()函数计算,简写$S(outy1)$,同理$T(w5)'=(1*w5^0)*h1=h1$

4.最后$E(w5)'=(outy1-o1)*S(outy1)*h1$



$E(w1)=h1'*w5*o1'*E(o1)'+h1'*w7*o2'*E(o2)'=(E(o1)'*w5*o1'+E(o2)'*w7*o2')*h1'$



1. $E(w5)'=z*Z(w5)'*U(w5)'$  , $Ez$

2. $Z(w5)'=[sigmoid(w5*outh1+b2)-o1]' = ((sigmoid_derivative(outy1)-0)$

3. $U(w5)'=1*outh1+0$

4. $E(w5)'=(outy1-o1)*sigmoid_derivative(outy1)*outh1$

x


https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.2.0%2Bcpu.zip

https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.2.0%2Bcpu.zip

D:\libtorch2.2.0\cpu
D:\libtorch2.2.0\debug




torch 张量常用创建方式

1. torch::tensor()   手动输入数据     
   auto input = torch::tensor({ 0.050,0.10 }, torch::kDouble);
   
2.    torch::randn()  随机初始化（正态）  标准正态分布（N (0,1)）
	  auto a = torch::randn({2, 2});	

3.   torch::rand() / torch::randint()
   
4.  torch::zeros() / torch::ones() / torch::full()  固定值填充   全 0、全 1 或指定常数

    auto x = torch::ones({2, 2});
	
5. 	torch::zeros_like() 基于现有张量创建   复用形状和属性，减少代码冗余

6.  torch::from_blob() 与 NumPy 互操作 共享内存，高效转换


torch 求导

自定义平方函数求导


在 PyTorch C++ API 中，若要对自定义函数（Function）实现自动求导，需要通过继承 torch::autograd::Function 并手动定义前向传播（forward）和反向传播（backward）逻辑。
这与 Python 中的 torch.autograd.Function 机制类似，核心是显式实现梯度计算规则。

基本步骤继承 torch::autograd::Function，并指定模板参数（输入 / 输出张量的类型）。实现静态 forward 函数：定义前向传播的计算逻辑。实现静态 backward 函数：根据链式法则，定义输入张量的梯度计算逻辑。封装为可调用接口：通过 apply 方法调用自定义函数。
示例：自定义平方函数求导假设我们要实现一个自定义函数 \(y = x^2\)，并手动定义其导数 \(\frac{dy}{dx} = 2x\)。

#include <torch/torch.h>
#include <iostream>

// 1. 继承 torch::autograd::Function，模板参数为输入和输出的类型（此处均为 Tensor）
class SquareFunction : public torch::autograd::Function<SquareFunction> {
public:
    // 2. 前向传播：计算 y = x^2
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,  // 上下文，用于保存反向传播需要的信息
        const torch::Tensor& x                  // 输入张量
    ) {
        // 保存输入 x 到上下文，供反向传播使用
        ctx->save_for_backward({x});
        // 前向计算：y = x^2
        return x * x;
    }

    // 3. 反向传播：计算输入 x 的梯度（dy/dx = 2x * 上游梯度）
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,  // 上下文，用于获取前向保存的信息
        torch::autograd::tensor_list grad_output  // 上游梯度（即 dy/dy'，其中 y' 是依赖 y 的下游变量）
    ) {
        // 从上下文获取前向保存的 x
        auto saved = ctx->get_saved_variables();
        torch::Tensor x = saved[0];

        // 上游梯度（通常是标量或与 y 同形状的张量，此处假设为 grad_output[0]）
        torch::Tensor grad_x = 2 * x * grad_output[0];

        // 返回输入 x 的梯度（若有多个输入，按顺序返回对应梯度）
        return {grad_x};
    }
};

// 4. 封装为方便调用的函数
torch::Tensor square(const torch::Tensor& x) {
    return SquareFunction::apply(x);
}

int main() {
    // 创建需要求导的输入张量
    torch::Tensor x = torch::tensor(3.0, torch::requires_grad(true));
    
    // 调用自定义函数：y = x^2
    torch::Tensor y = square(x);
    
    // 反向传播（求 y 对 x 的导数）
    y.backward();
    
    // 输出结果：x=3 时，dy/dx=2*3=6
    std::cout << "x.grad() = " << x.grad() << std::endl;  // 输出：6.0

    return 0;
}