# ZeroBaseLearnAI
learn ai 
https://zhuanlan.zhihu.com/p/1892169407620444883



------------------------------------------------------------------------

输入层： 两个点
i1 = 0.05;
i2 = 0.1;

------------------------------------------------------------------------


隐藏层 两个点
 h1 = ?;  加权求和后的值 ;  sigmoid 激活后的值 outh1 = ?;
 h2 = ?;  加权求和后的值  ; sigmoid 激活后的值  outh2 = ?;

输入层到隐藏层 h1点 权重
  w1 = 0.15;
  w2 = 0.2;
  
输入层到隐藏层 h2点 权重
  w3 = 0.25;
  w4 = 0.3; 
  
偏置
 b1 = 0.35; 


h1 = i1 * w1 + i2 * w2 + b1
h2 = i1 * w3 + i2 * w4 + b1

outh1 = sigmoid(h1)
outh2 = sigmoid(h2)

------------------------------------------------------------------------

输出层 两个点

 y1 = ?;  加权求和后的值  ; sigmoid 激活后的值 outy1 = ?
 y2 = ?;  加权求和后的值  ; sigmoid 激活后的值 outy2 = ?

隐藏层到 y1点 权重
  w5 = 0.4;
  w6 = 0.45;
  
隐藏层到 y2点 权重
  w7 = 0.5;
  w8 = 0.55; 
  
偏置
 b2 = 0.6;

y1 = outh1 * w5 + outh2 * w6 + b2
y2 = outh1 * w7 + outh2 * w8 + b2

outy1 = sigmoid(y1)
outy2 = sigmoid(y2)

------------------------------------------------------------------------

损失函数 E

E =  0.5 * (outy1 - o1)^2 + 0.5 * (outy2 - o2)^2

E对w5的导数，相关函数关系如下    （函数推导）

y1 = outh1 * w5 + outh2 * w6 + b2  ->    y1(outh1) = outh1 * w5 + outh2 * w6 + b2
outy1 = sigmoid(y1)                -->   outy1(y1) = sigmoid(y1) 

E =  0.5 * (outy1 - o1)^2 + 0.5 * (outy2 - o2)^2  -->  E(outy1,outy2) = 0.5 * (outy1 - o1)^2 + 0.5 * (outy2 - o2)^2 

对w5求偏导， 
outh2
w6
b2
outy2
o2
都是已知值 常量的导数为0，简化表达示

 y1(outh1) = outh1 * w5
 outy1(y1) = sigmoid(y1)
 E(outy1) = 0.5 * (outy1 - o1)^2 


复合函数求导（链式法则）
若 (y = f(u)) 且 (u = g(x))， 则 f′(u)*g′(x)
E(w5)' =  E(outy1)' * outy1(y1)'* y1(w5)'
E(outy1)' = outy1 - o1 
outy1(y1)' = sigmoid(outy1)' 
y1(w5)' = outh1 * (w5 == 1)

E(w5)' = (outy1 - o1)* sigmoid_derivative_from_sigmoid(outy1) * outh1


///  w1的导数
h1 = i1 * w1 + i2 * w2 + b1
outh1 = sigmoid(h1)
y1 = outh1 * w5 + outh2 * w6 + b2
outy1 = sigmoid(y1)
y2 = outh1*w7 + outh2 * w8 + b2
outy2 = sigmoid(y2)

E =  0.5 * (outy1 - o1)^2 + 0.5 * (outy2 - o2)^2

常量的导数为0，简化表达示
h1 = i1 * w1
--------------------------
outh1 = sigmoid(h1)
y1 = outh1 * w5 + outh2 * w6 + b2
---> y1= outh1 * w5 

y2 = outh1 * w7 + outh2 * w8 + b2
 ---> y2 = outh1 * w7
--------------------------
outy1 = sigmoid(y1)
outy2 = sigmoid(y2)
E =  0.5*(outy1 - o1)^2 +  0.5*(outy2 - o2)^2
---------------------------------------------------------------------------------

h1 = i1 * w1                h1(i1) = i1 * w1
outh1 = sigmoid(h1)         outh1(h1) = sigmoid(h1)
y1= outh1 * w5              y1(outh1) = outh1 * w5
y2 = outh1 * w7 ；          y2(outh1) = outh1 * w7
outy1 = sigmoid(y1)         outy1(y1) = sigmoid(y1) 
outy2 = sigmoid(y2)	        outy2(y2) =  sigmoid(y2)
							eouty1(outy1) = 0.5*(outy1 - o1)^2
							eouty2(outy2) = 0.5*(outy2 - o2)^2
						    E(eouty1,eouty2) = eouty1 + eouty2 
							
1. E(w1)' = E(eouty1,eouty2)'	
	
2. E(eouty1,eouty2)' = eouty1(outy1)' + eouty2(outy2)' 

3. eouty1(outy1)' = (outy1 - o1)*sigmoid_derivative_from_sigmoid(outy1)*sigmoid_derivative_from_sigmoid(outh1)*w5*i1

4. eouty1(outy2)' = (outy2 - o2)*sigmoid_derivative_from_sigmoid(outy2)*sigmoid_derivative_from_sigmoid(outh1)*w7*i1

5. E(w1)' = (outy1 - o1)*sigmoid_derivative_from_sigmoid(outy1)*sigmoid_derivative_from_sigmoid(outh1)*w5*i1 +  (outy2 - o2)*sigmoid_derivative_from_sigmoid(outy2)*sigmoid_derivative_from_sigmoid(outh1)*w7*i1

6. E(w1)' = ((outy1 - o1)*sigmoid_derivative_from_sigmoid(outy1)*w5 + (outy2 - o2)*sigmoid_derivative_from_sigmoid(outy2)*w7)*sigmoid_derivative_from_sigmoid(outh1)*i1

