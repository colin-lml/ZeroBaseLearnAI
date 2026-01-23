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



00000000
00011000
00011000
00011000
00011000
00011000
00011000
00000000

00000000
01111100
01111100
00001100
00001100
00001100
00001100
00000000

卷积核就是图像处理时，给定输入图像，输入图像中一个小区域中像素加权平均后成为输出图像中的每个对应像素，其中权值由一个函数定义，这个函数称为卷积核。




高斯模糊
边缘检测
锐化‌
浮雕效果

卷积概念
卷积在图像处理中主要用于特征提取、滤波和模式识别，通过卷积核（滤波器）对图像局部区域进行加权计算，生成特征图。

高斯模糊
边缘检测
锐化
浮雕效果


图片效果图




一维卷积原理
举例子来说明，假设有输入序列数据 [1, 2, 3, 4, 5] 和 一组权重数据 [1, 0, 1],做以下运算
1. [1, 2, 3] * [1, 0, 1] =  1*1+2*0+3*1 = 4
2. [2, 3, 4] * [1, 0, 1] = 2*1+3*0+4*1 = 6
3. [3, 4, 5] * [1, 0, 1] = 3*1+4*0+5*1 = 8 
输出结果：[4, 6, 8]
像这样的计算叫卷积运算
[1, 0, 1]叫卷积核,长度==3

[1, 2, 3, 4, 5]
[1, 0, 1]

[1, 2, 3, 4, 5]
   [1, 0, 1]

[1, 2, 3, 4, 5]
      [1, 0, 1]

1. 如果想要输出结果与输入数据长度一样，在原数据左右两边填充0 如下  
[0,1, 2, 3, 4, 5,0]  与 [1, 0, 1]做以下运算
1. [0, 1, 2] * [1, 0, 1] =  0*1+2*0+2*1 = 2
2. [1, 2, 3] * [1, 0, 1] =  1*1+2*0+3*1 = 4
3. [2, 3, 4] * [1, 0, 1] = 2*1+3*0+4*1 = 6
4. [3, 4, 5] * [1, 0, 1] = 3*1+4*0+5*1 = 8  
4. [4, 5, 0] * [1, 0, 1] = 4*1+5*0+0*1 = 4 
输出结果：[2, 4, 6, 8, 4]
把这个叫填充一般是填充0

2. 以上计算卷积核在原数据上一次只移动一个数据位其步长为1，如步长为2时
1. [1, 2, 3] * [1, 0, 1] =  1*1+2*0+3*1 = 4
2. [3, 4, 5] * [1, 0, 1] = 3*1+4*0+5*1 = 8 
输出结果：[4, 8]

总结：
卷积运算过程：卷积核从左往右滑动加权求和
[1, 2, 3, 4, 5]
[1, 0, 1]
[1, 2, 3, 4, 5]
   [1, 0, 1]
[1, 2, 3, 4, 5]
      [1, 0, 1]
	  
inlength：原数据长度
padding：填充位
kernelsize：卷积核大小
stride:步长
输出长度： (inlength  +  2*padding - (kernelsize-1) -1)/stride + 1

卷积





假设输入序列为 [1, 2, 3, 4, 5]（长度 = 5），卷积核为 [0.2, 0.5, 0.3]（kernel_size=3），stride=1，padding=0
卷积核覆盖 [1, 2, 3] → 计算：1*0.2 + 2*0.5 + 3*0.3 = 0.2 + 1 + 0.9 = 2.1
滑动 1 步，覆盖 [2, 3, 4] → 计算：2*0.2 + 3*0.5 + 4*0.3 = 0.4 + 1.5 + 1.2 = 3.1
滑动 1 步，覆盖 [3, 4, 5] → 计算：3*0.2 + 4*0.5 + 5*0.3 = 0.6 + 2 + 1.5 = 4.1
输出序列为 [2.1, 3.1, 4.1]（长度 = 3）。

out_length = floor((in_length + 2*padding - (kernel_size - 1) - 1) / stride + 1)

二维卷积原理
二维卷积运算过程和一维卷积一样它处理矩阵数据用举例子来说明
输入矩阵数据：
0,0,0,0,0,0,0,0,0
0,1, 2, 3, 4, 5,0
0,1, 2, 3, 4, 5,0
0,1, 2, 3, 4, 5,0
0,1, 2, 3, 4, 5,0
0,1, 2, 3, 4, 5,0
0,0,0,0,0,0,0,0,0

1, 0, 1
1, 0, 1
1, 0, 1

(1*1+2*0+3*1) + (1*1+2*0+3*1) + (1*1+2*0+3*1) = 12

卷积神经网络的中

池化层

池化是一种用于降维、提取局部显著特征的操作,常用方式
1.最大池化， 在池化窗口内取最大值作为输出，示例： 窗口 [2, 5, 3] ，输出 5。
2.平均池化，在池化窗口内取平均值作为输出，平滑局部特征
3.随机池化，在池化窗口内随机采样



卷积神经网络

在图像处理中卷积核是已知，使用不同的卷积核提取多个维度的特征图，在卷积神经网络中卷积核是未知的(随机值)，
构建一个模型把它训练出来跟神经网络求权重方式一样。卷积神经网络的模型从它命名可以看出来它是由卷积层加一个神经网络
一简单卷积神经网络结构:
1.卷积层  一个卷积核运算(提取特征)
2.激活函数 
3. 池化层
4.全连接层 将二维数据转换成一维向量数据(神经网络输入层-> 神经网络输出层)

复杂卷积神经网络结构
1.卷积层
2.激活函数
3. 池化层
4.卷积层
5.激活函数
6. 池化层
.....
全连接层
神经网络隐藏层1
神经网络隐藏层2
...
神经网络输出层

适用场景
图像分类、目标检测、语义分割、图像生成、人脸识别等，能自动捕捉图像的边缘、纹理、形状等层级特征。



transformer

transformer原本是聚焦在自然语言处理领域，

在机器翻译任务上，Transformer表现超过了RNN和CNN，只需要编/解码器就能达到很好的效果，可以高效地并行化。

Transformer模型提出了自注意力机制（Self-attention），是一个重大改进。


Softmax 是一个归一化函数，它将一个实数向量转换成一个概率分布（每个值都在 0～1 之间，所有值加起来等于 1）



https://arxiv.org/pdf/1706.03762


Transformer模型架构是一种深度学习模型，由谷歌大脑团队Ashish Vaswani等人于2017年在论文《Attention is All You Need》中提出，采用自注意力机制替代循环神经网络和卷积神经网络，实现自然语言处理任务的并行计算

Transformer模型架构是由谷歌大脑团队在2017年提出论文，

这篇文档是 Transformer 模型的原始论文，核心是提出了一种完全靠 “注意力机制” 工作的神经网络，不用之前常用的循环或卷积结构，简单说就是：
之前做翻译、文本处理的模型，大多靠循环网络一步步处理数据，慢还难并行；
而 Transformer 全靠注意力机制，能同时处理整个序列，训练速度快很多，还能捕捉长距离的语义关联。
核心是 “多头注意力”：就像人看书时会同时关注不同重点，模型会分成多个 “注意力头”，各自关注序列里不同位置的信息，再把结果整合，能更全面地理解数据。还有个 “缩放点积” 的技巧，避免计算时数值太大影响效果。
因为没有循环结构，模型不知道单词的顺序，所以加了 “位置编码”：用正弦余弦函数给每个位置的单词加个 “位置标签”，让模型知道谁在前谁在后。
模型分编码器和解码器：编码器处理输入（比如要翻译的原文），解码器生成输出（比如翻译后的译文），中间靠注意力机制连接，解码器还会 “屏蔽未来位置”，避免提前偷看还没生成的内容。
实验效果很牛：在英德、英法翻译任务上，比当时最好的模型效果还好，训练时间却短很多（比如英法翻译用 8 块 GPU 训 3.5 天就达到新纪录）；还能用到其他任务，比如英语语法分析，就算训练数据不多也能有好效果。
简单总结就是：用注意力机制替代传统循环 / 卷积，让模型又快又强，不仅能做好翻译，还能适配多种文本处理任务，后来成了大语言模型的核心基础。



论文指出在做翻译、文本处理时，大多靠循环网络一步步处理数据，慢还难并行，Transformer 靠注意力机制，能同时处理整个序列，训练速度快很多，还能捕捉长距离的语义关联
Transformer没有循环结构，模型不知道单词的顺序，所以加了 “位置编码”,用正弦余弦函数给每个位置的单词加个 “位置标签”，让模型知道谁在前谁在后。
简单总结就是：用注意力机制替代传统循环 / 卷积，让模型又快又强，不仅能做好翻译，还能适配多种文本处理任务，后来成了大语言模型的核心基础。



注意力机制 就像人看书时会同时关注不同重点，模型会分成多个 “注意力头”，各自关注序列里不同位置的信息，再把结果整合，能更全面地理解数据。



前馈网络

TransformerEncoderLayer：单个编码层的实现
TransformerEncoder：编码层的串联组织
TransformerDecoderLayer：单个解码层的实现
TransformerDecoder：解码层的串联组织


一个是残差结构，一个是LayerNorm

Transformer解析结构先把重复的部份删除，列举一下主要结构组件：
1. input/output Embedding                       词嵌入 
2. Positional Encoding                          位置编码
3. Multi-Head/Masked-Multi-Head  Attention      注意力层
4. Add & norm                                   残差结构,归一化 
5. Feed Forward                                 前馈网络
6. linear 									    全连接层
7. Softmax  								    逻辑回归函数   



⾃然语⾔是⼀套⽤来表达含义的复杂系统。在这套系统中，词是表义的基本单元。顾名思义，词向量是⽤来表⽰词的向量，也可被认为是词的特征向量或表征。把词映射为实数域向量的技术也叫词嵌⼊（word embedding）。近年来，词嵌⼊已逐渐成为⾃然语⾔处理的基础知识。

在NLP(自然语言处理)领域，文本表示是第一步，也是很重要的一步，通俗来说就是把人类的语言符号转化为机器能够进行计算的数字，因为普通的文本语言机器是看不懂的，必须通过转化来表征对应文本。早期是基于规则的方法进行转化，而现代的方法是基于统计机器学习的方法。

数据决定了机器学习的上限,而算法只是尽可能逼近这个上限，在本文中数据指的就是文本表示，所以，弄懂文本表示的发展历程，对于NLP学习者来说是必不可少的。接下来开始我们的发展历程。文本表示分为离散表示和分布式表示：



One-hot简称读热向量编码，也是特征工程中最常用的方法。

随着语料库的增加，数据特征的维度会越来越大，产生一个维度很高，又很稀疏的矩阵。
这种表示方法的分词顺序和在句子中的顺序是无关的，不能保留词与词之间的关系信息。


词袋模型


词袋模型(Bag-of-words model)，像是句子或是文件这样的文字可以用一个袋子装着这些词的方式表现，这种表现方式不考虑文法以及词的顺序。


TF-IDF

TF-IDF（term frequency–inverse document frequency）是一种用于信息检索与数据挖掘的常用加权技术。TF意思是词频(Term Frequency)，IDF意思是逆文本频率指数(Inverse Document Frequency)。

字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章

n-gram模型

n-gram模型为了保持词的顺序，做了一个滑窗的操作，这里的n表示的就是滑窗的大小，例如2-gram模型，也就是把2个词当做一组来处理，然后向后移动一个词的长度，再次组成另一组词，把这些生成一个字典，按照词袋模型的方式进行编码得到结果。改模型考虑了词的顺序

 离散表示存在的问题
 
 由于存在以下的问题，对于一般的NLP问题，是可以使用离散表示文本信息来解决问题的，但对于要求精度较高的场景就不适合了。

无法衡量词向量之间的关系。
词表的维度随着语料库的增长而膨胀。
n-gram词序列随语料库增长呈指数型膨胀，更加快。
离散数据来表示文本会带来数据稀疏问题，导致丢失了信息，与我们生活中理解的信息是不一样的


3. 分布式表示

共现矩阵

神经网络表示


Word2Vec
谷歌2013年提出的Word2Vec是目前最常用的词嵌入模型之一。Word2Vec实际是一种浅层的神经网络模型，它有两种网络结构，分别是CBOW（Continues Bag of Words）连续词袋和Skip-gram。Word2Vec和上面的NNLM很类似，但比NNLM简




通俗点来说，PyTorch中的Embedding技术，就像是一本巨大的字典，其中每个单词都对应一个数字列表（向量）。这种技术帮助计算机理解单词之间的关系，就像我们通过单词的使用上下文来理解它们的意义一样。在处理文本或语言数据时，Embedding可以将简单的单词转换成计算机能够处理的数值形式，让计算机能够更好地学习和理解自然语言。


// -------------------------- 1. 基础嵌入层 --------------------------
    // 定义嵌入层：词汇表大小=10，嵌入维度=5
    torch::nn::Embedding embed(torch::nn::EmbeddingOptions(10, 5));
    
    // 初始化嵌入权重（推荐用xavier_uniform，和Transformer保持一致）
    torch::nn::init::xavier_uniform_(embed->weight);
    
    // 输入：形状为 [batch_size, seq_len] 的整数索引（比如2个样本，每个样本3个单词ID）
    torch::Tensor input = torch::tensor({
        {1, 3, 5},
        {2, 4, 6}
    }, torch::kLong);
    
    // 前向传播：输出形状 [batch_size, seq_len, embedding_dim] = [2,3,5]
    torch::Tensor output = embed(input);
    
    std::cout << "=== 基础嵌入层示例 ===" << std::endl;
    std::cout << "输入形状: " << input.sizes() << std::endl;
    std::cout << "输出形状: " << output.sizes() << std::endl;
    std::cout << "第一个样本的嵌入向量:\n" << output[0] << "\n" << std::endl;

    // -------------------------- 2. 带padding_idx的嵌入层 --------------------------
    // 定义嵌入层：指定padding_idx=0（ID=0的向量始终为0）
    torch::nn::Embedding embed_pad(torch::nn::EmbeddingOptions(10, 5).padding_idx(0));
    torch::nn::init::xavier_uniform_(embed_pad->weight);
    
    // 输入包含padding（ID=0）
    torch::Tensor input_pad = torch::tensor({
        {0, 1, 0},
        {3, 0, 5}
    }, torch::kLong);
    torch::Tensor output_pad = embed_pad(input_pad);
    
    std::cout << "=== 带padding的嵌入层 ===" << std::endl;
    std::cout << "padding位置的向量（全0）:\n" << output_pad[0][0] << "\n" << std::endl;

    // -------------------------- 3. Transformer风格的维度调整 --------------------------
    // Transformer要求输入维度为 [seq_len, batch_size]（序列长度在前）
    torch::Tensor input_trans = input.transpose(0, 1);  // [3, 2]
    torch::Tensor output_trans = embed(input_trans);    // [3, 2, 5]
    
    std::cout << "=== Transformer风格维度 ===" << std::endl;
    std::cout << "Transformer输入形状: " << input_trans.sizes() << std::endl;
    std::cout << "Transformer嵌入输出形状: " << output_trans.sizes() << std::endl;





#include <torch/torch.h>
#include <iostream>
#include <iomanip>  // 用于格式化输出

// 1. 定义包含可训练 Embedding 层的模型（复用基础结构）
struct EmbeddingModel : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};

    // 构造函数：初始化 Embedding 层（保留可训练特性）
    EmbeddingModel(int64_t num_embeddings, int64_t embedding_dim) {
        embedding = register_module(
            "embedding",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                .padding_idx(-1)  // 禁用padding，所有索引参与训练
                .max_norm(1.0)     // 限制向量范数，防止训练发散
            )
        );
    }

    // 前向传播：输出嵌入向量
    torch::Tensor forward(torch::Tensor x) {
        return embedding->forward(x);
    }
};

int main() {
    // 2. 训练超参数配置
    const int64_t num_embeddings = 10;     // 小词典大小（便于观察参数变化）
    const int64_t embedding_dim = 4;       // 小嵌入维度（简化计算）
    const int64_t batch_size = 2;          // 批次大小
    const int64_t seq_len = 3;             // 序列长度
    const int64_t epochs = 50;             // 训练轮数
    const float lr = 0.1;                  // 学习率

    // 3. 初始化模型、损失函数、优化器
    EmbeddingModel model(num_embeddings, embedding_dim);
    // 损失函数：MSELoss（适合回归类任务，衡量嵌入输出与目标的差距）
    torch::nn::MSELoss loss_fn;
    // 优化器：SGD（绑定模型参数，设置学习率）
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(lr));

    // 4. 生成固定的训练数据（输入索引 + 目标嵌入向量）
    // 输入索引：[batch_size, seq_len]，类型必须为kLong
    torch::Tensor input_indices = torch::tensor({
        {1, 3, 5},  // 样本1的索引
        {2, 4, 6}   // 样本2的索引
    }, torch::kLong);
    // 目标嵌入向量：[batch_size, seq_len, embedding_dim]（模型要拟合的目标）
    torch::Tensor target_embeddings = torch::tensor({
        {{0.1, 0.2, 0.3, 0.4}, {0.3, 0.4, 0.5, 0.6}, {0.5, 0.6, 0.7, 0.8}},  // 样本1目标
        {{0.2, 0.3, 0.4, 0.5}, {0.4, 0.5, 0.6, 0.7}, {0.6, 0.7, 0.8, 0.9}}   // 样本2目标
    }, torch::kFloat);

    // 打印初始嵌入矩阵（训练前的参数）
    std::cout << "=== 训练前嵌入矩阵 ===" << std::endl;
    auto init_embedding = model.embedding->weight.data();
    std::cout << init_embedding << std::endl;

    // 5. 核心训练循环
    std::cout << "\n=== 开始训练 ===" << std::endl;
    model.train();  // 设置模型为训练模式（Embedding层无dropout/bn，仅规范操作）
    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        // 步骤1：清空梯度（必须！否则梯度会累积）
        optimizer.zero_grad();

        // 步骤2：前向传播
        torch::Tensor output = model.forward(input_indices);

        // 步骤3：计算损失（输出与目标的MSE）
        torch::Tensor loss = loss_fn(output, target_embeddings);

        // 步骤4：反向传播（计算梯度，包括Embedding层的weight梯度）
        loss.backward();

        // 步骤5：优化器更新参数（更新Embedding层的嵌入矩阵）
        optimizer.step();

        // 打印训练进度（每5轮打印一次）
        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch: " << std::setw(3) << (epoch + 1) 
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss.item<float>() << std::endl;
        }
    }

    // 6. 打印训练后嵌入矩阵（对比训练前的变化）
    std::cout << "\n=== 训练后嵌入矩阵 ===" << std::endl;
    auto final_embedding = model.embedding->weight.data();
    std::cout << final_embedding << std::endl;

    // 7. 验证训练效果：用训练后的模型推理
    std::cout << "\n=== 训练后推理示例 ===" << std::endl;
    torch::Tensor test_input = torch::tensor({1, 3}, torch::kLong);  // 测试索引1和3
    torch::Tensor test_output = model.forward(test_input);
    std::cout << "测试输入索引: [1, 3]" << std::endl;
    std::cout << "训练后嵌入输出:\n" << test_output << std::endl;

    return 0;
}




