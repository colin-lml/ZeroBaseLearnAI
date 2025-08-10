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
