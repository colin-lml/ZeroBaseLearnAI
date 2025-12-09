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


AI 基础知识六 libtorch构建卷积神经网络




AI 基础知识六 libtorch构建卷积神经网络

AI 基础知识七 数据集与循环神经网络RNN

torch::data::datasets::Dataset 

抽象基类，定义数据集的 “接口规范”
1. 必须重写 size()（返回总样本数）和 get()（返回单样本）；
2. 支持 CRTP 模式（Dataset<Self>），实现类型推导；
3. 是 DataLoader 加载数据的唯一数据源。

torch::data::Example
单样本容器，封装 “数据 + 标签”（或纯数据）

1. 模板类，默认是 Example<Tensor, Tensor>（数据 + 标签）；
2. 可自定义为单张量（Example<Tensor>）或多标签（Example<Tensor, std::tuple<Tensor, Tensor>>）；
3. 是 Dataset::get() 的返回类型，也是 DataLoader 拼接批量的最小单元。


Dataset 生产 Example：
Dataset 的 get(size_t index) 方法必须返回一个 Example 对象，即 “每个样本都被封装为 Example”；
Example 是批量拼接的基础：
DataLoader 收集多个 Example，通过 Stack 变换将其中的 data 和 target 分别拼接为批量张量（[batch_size, ...]）；
类型对齐：
Dataset 的模板参数（如 Dataset<MyDataset>）需与 Example 的类型匹配，确保 DataLoader 能正确推导批量张量的类型。


数据加载器（DataLoader）

torch::data::make_data_loader 是 LibTorch 中创建数据加载器（DataLoader）的核心函数，负责将 Dataset 封装为可迭代的批量数据生成器，实现「单样本→批量数据」的转换，同时支持多线程加载、数据打乱、批量大小配置等工业级特性




#include <torch/torch.h>
#include <iostream>
#include <vector>

// ===================== 1. 定义Embedding+RNN模型 =====================
class EmbeddingRNNClassifier : public torch::nn::Module {
public:
    // 构造函数：初始化嵌入层、RNN层、分类头
    // vocab_size: 词汇表大小（离散单词ID总数）
    // embed_dim: 词向量维度
    // hidden_size: RNN隐藏层维度
    // num_classes: 分类类别数
    EmbeddingRNNClassifier(
        int64_t vocab_size, 
        int64_t embed_dim, 
        int64_t hidden_size, 
        int64_t num_classes
    ) : 
        embedding_(torch::nn::EmbeddingOptions(vocab_size, embed_dim).padding_idx(0)), // 0为填充位
        rnn_(torch::nn::RNNOptions(embed_dim, hidden_size)
             .num_layers(1)          // 单层RNN
             .batch_first(true)      // 输入格式：(batch_size, seq_len, embed_dim)（更直观）
             .bidirectional(false)), // 单向RNN
        fc_(hidden_size, num_classes) { // 分类全连接层
        
        // 注册子模块（必须！否则参数无法被优化器捕获）
        register_module("embedding", embedding_);
        register_module("rnn", rnn_);
        register_module("fc", fc_);
    }

    // 前向传播
    torch::Tensor forward(torch::Tensor x) {
        // x: (batch_size, seq_len) → 输入为单词ID序列
        
        // Step 1: Embedding层 → 词向量序列
        // output: (batch_size, seq_len, embed_dim)
        torch::Tensor embed = embedding_->forward(x);
        
        // Step 2: RNN层提取时序特征
        // 初始化隐藏状态h0: (num_layers * num_directions, batch_size, hidden_size)
        auto h0 = torch::zeros({1, x.size(0), rnn_->options.hidden_size()}, 
                               torch::device(embed.device()).dtype(embed.dtype()));
        // RNN输出：(output, hn)
        // output: (batch_size, seq_len, hidden_size) → 所有时间步输出
        // hn: (1, batch_size, hidden_size) → 最后时刻隐藏状态
        auto rnn_out = rnn_->forward(embed, h0);
        torch::Tensor hn = std::get<1>(rnn_out); // 提取最后时刻隐藏状态
        
        // Step 3: 分类头（去掉num_layers维度）
        // hn.squeeze(0): (batch_size, hidden_size)
        torch::Tensor logits = fc_(hn.squeeze(0)); // 输出：(batch_size, num_classes)
        
        return logits;
    }

private:
    torch::nn::Embedding embedding_; // 嵌入层
    torch::nn::RNN rnn_;             // RNN层
    torch::nn::Linear fc_;           // 分类全连接层
};

// ===================== 2. 生成模拟文本序列数据 =====================
// 生成：(batch_size, seq_len)的单词ID序列 + (batch_size,)的标签
std::pair<torch::Tensor, torch::Tensor> generate_text_data(
    int64_t batch_size, 
    int64_t seq_len, 
    int64_t vocab_size, 
    int64_t num_classes
) {
    // 单词ID序列：值范围0~vocab_size-1（0为填充位）
    torch::Tensor input_ids = torch::randint(0, vocab_size, {batch_size, seq_len}, torch::kLong);
    // 分类标签：0/1（二分类）
    torch::Tensor labels = torch::randint(0, num_classes, {batch_size}, torch::kLong);
    return {input_ids, labels};
}

// ===================== 3. 主函数（训练+预测） =====================
void RnnMain() {
    // -------------------- 超参数设置 --------------------
    const int64_t vocab_size = 1000;    // 词汇表大小（单词ID：0~999）
    const int64_t embed_dim = 64;       // 词向量维度
    const int64_t hidden_size = 128;    // RNN隐藏层维度
    const int64_t num_classes = 2;      // 二分类
    const int64_t seq_len = 15;         // 序列长度（每个文本15个单词）
    const int64_t batch_size = 16;      // 批次大小
    const int64_t epochs = 30;          // 训练轮数
    const float lr = 0.001f;            // 学习率

    // -------------------- 初始化模型/优化器/损失函数 --------------------
    EmbeddingRNNClassifier model(vocab_size, embed_dim, hidden_size, num_classes);
    // 优化器：Adam（适配嵌入层+RNN的参数更新）
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lr));
    // 损失函数：交叉熵（适配分类任务）
    torch::nn::CrossEntropyLoss criterion;

    // -------------------- 训练循环 --------------------
    model.train(); // 训练模式
    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        // 生成一批训练数据
        auto [input_ids, labels] = generate_text_data(batch_size, seq_len, vocab_size, num_classes);
        
        // 前向传播
        optimizer.zero_grad(); // 梯度清零
        torch::Tensor logits = model.forward(input_ids);
        
        // 计算损失
        torch::Tensor loss = criterion(logits, labels);
        
        // 反向传播 + 更新参数
        loss.backward();
        optimizer.step();

        // 打印训练信息（每5轮）
        if ((epoch + 1) % 5 == 0) {
            // 计算准确率
            auto preds = logits.argmax(1); // 预测类别：(batch_size,)
            float acc = preds.eq(labels).sum().item<float>() / batch_size;
            
            std::cout << "Epoch: " << epoch + 1 
                      << " | Loss: " << loss.item<float>() 
                      << " | Acc: " << acc << std::endl;
        }
    }

    // -------------------- 预测示例 --------------------
    model.eval(); // 评估模式
    torch::NoGradGuard no_grad; // 禁用梯度计算（提升推理效率）

    // 生成单个测试样本（batch_size=1）
    auto [test_ids, test_label] = generate_text_data(1, seq_len, vocab_size, num_classes);
    torch::Tensor test_logits = model.forward(test_ids);
    auto pred_label = test_logits.argmax(1).item<int64_t>();
    auto true_label = test_label.item<int64_t>();

    std::cout << "\n=== 预测结果 ===" << std::endl;
    std::cout << "输入单词ID序列:\n" << test_ids.squeeze(0) << std::endl;
    std::cout << "真实标签: " << true_label << std::endl;
    std::cout << "预测标签: " << pred_label << std::endl;

   
}





import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 数据集：字符序列预测（Hello -> Elloh）
char_set = list("hello")
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

# 数据准备
input_str = "hello"
target_str = "elloh"
input_data = [char_to_idx[c] for c in input_str]
target_data = [char_to_idx[c] for c in target_str]

# 转换为独热编码
input_one_hot = np.eye(len(char_set))[input_data]

# 转换为 PyTorch Tensor
inputs = torch.tensor(input_one_hot, dtype=torch.float32)
targets = torch.tensor(target_data, dtype=torch.long)

# 模型超参数
input_size = len(char_set)
hidden_size = 8
output_size = len(char_set)
num_epochs = 200
learning_rate = 0.1

# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)  # 应用全连接层
        return out, hidden

model = RNNModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练 RNN
losses = []
hidden = None  # 初始隐藏状态为 None
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 前向传播
    outputs, hidden = model(inputs.unsqueeze(0), hidden)
    hidden = hidden.detach()  # 防止梯度爆炸

    # 计算损失
    loss = criterion(outputs.view(-1, output_size), targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试 RNN
with torch.no_grad():
    test_hidden = None
    test_output, _ = model(inputs.unsqueeze(0), test_hidden)
    predicted = torch.argmax(test_output, dim=2).squeeze().numpy()

    print("Input sequence: ", ''.join([idx_to_char[i] for i in input_data]))
    print("Predicted sequence: ", ''.join([idx_to_char[i] for i in predicted]))
	
	
	
	
	
	
	
	
/*******************************************************************************************************/	
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// ===================== 1. 定义正弦波数据集（序列数据） =====================
// 输入：前 seq_len 个正弦值，输出：下一个正弦值
class SineDataset : public torch::data::Dataset<SineDataset> {
public:
    SineDataset(int seq_len = 3, int total_samples = 1000) 
        : seq_len_(seq_len) {
        // 生成正弦波数据
        for (int i = 0; i < total_samples + seq_len; ++i) {
            data_.push_back(std::sin(0.1f * i)); // 0.1*i 控制正弦波周期
        }
    }

    // 返回数据集总样本数
    size_t size() const override {
        return data_.size() - seq_len_;
    }

    // 返回单个样本（输入序列 + 目标值）
    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override {
        // 构造输入序列：[seq_len, 1]（单特征序列）
        std::vector<float> input_data(data_.begin() + index, data_.begin() + index + seq_len_);
        torch::Tensor input = torch::tensor(input_data)
            .reshape({seq_len_, 1})  // [seq_len, input_size=1]
            .to(torch::kFloat32);

        // 构造目标值：单个浮点值
        float target_val = data_[index + seq_len_];
        torch::Tensor target = torch::tensor({target_val}).to(torch::kFloat32);

        return {input, target};
    }

private:
    std::vector<float> data_;  // 原始正弦波数据
    int seq_len_;              // 输入序列长度
};

// ===================== 2. 定义极简 RNN 模型 =====================
class SimpleRNN : public torch::nn::Module {
public:
    SimpleRNN(int input_size = 1, int hidden_size = 16, int output_size = 1) 
        : hidden_size_(hidden_size) {
        // 配置 RNN 层参数
        torch::nn::RNNOptions rnn_options(input_size, hidden_size);
        rnn_options.batch_first(true);  // 输入格式：[batch_size, seq_len, input_size]

        // 注册 RNN 层和全连接层
        rnn_ = register_module("rnn", torch::nn::RNN(rnn_options));
        fc_ = register_module("fc", torch::nn::Linear(hidden_size, output_size));
    }

    // 前向传播：输入 x → [batch_size, seq_len, input_size]
    torch::Tensor forward(const torch::Tensor& x) {
        // RNN 输出：
        // outputs: [batch_size, seq_len, hidden_size]（所有时间步隐藏状态）
        // hidden: [1, batch_size, hidden_size]（最后一个时间步隐藏状态）
        auto [outputs, hidden] = rnn_->forward(x);

        // 取最后一个时间步的隐藏状态，输入全连接层预测
        torch::Tensor last_hidden = hidden.squeeze(0); // 去掉维度1 → [batch_size, hidden_size]
        return fc_->forward(last_hidden);
    }

private:
    torch::nn::RNN rnn_;          // RNN 层
    torch::nn::Linear fc_;        // 全连接层（映射隐藏状态到预测值）
    int hidden_size_;             // 隐藏层维度
};

// ===================== 3. 训练 + 推理主逻辑 =====================
void RnnMain
{
    // 超参数配置（极简版）
    const int seq_len = 3;        // 输入序列长度
    const int batch_size = 32;    // 批量大小
    const float lr = 1e-3;        // 学习率
    const int epochs = 30;        // 训练轮数

    // -------------------- 初始化数据集和数据加载器 --------------------
    auto dataset = SineDataset(seq_len, 1000);
    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).shuffle(true)
    );

    // -------------------- 初始化模型、优化器、损失函数 --------------------
    SimpleRNN model;              // 初始化 RNN 模型
    torch::optim::Adam optimizer(model->parameters(), lr); // Adam 优化器
    torch::nn::MSELoss criterion; // 均方误差损失（回归任务）

    // -------------------- 训练循环 --------------------
    model->train(); // 训练模式
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int batch_count = 0;

        for (auto& batch : *data_loader) {
            auto [x, y] = batch;  // x: [32, 3, 1], y: [32, 1]

            // 前向传播 + 计算损失
            auto y_pred = model->forward(x);
            auto loss = criterion(y_pred, y);

            // 反向传播 + 优化
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;
        }

        // 每5轮打印一次平均损失
        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " | Avg Loss: " << total_loss / batch_count << std::endl;
        }
    }

    // -------------------- 推理测试 --------------------
    model->eval(); // 推理模式（禁用 Dropout 等训练特化逻辑）
    std::cout << "\n=== 推理测试 ===" << std::endl;

    // 构造测试序列（未参与训练的新数据）
    std::vector<float> test_seq = {
        std::sin(0.1f * 1000),
        std::sin(0.1f * 1001),
        std::sin(0.1f * 1002)
    };
    torch::Tensor test_x = torch::tensor(test_seq)
        .reshape({1, seq_len, 1})  // [batch_size=1, seq_len=3, input_size=1]
        .to(torch::kFloat32);

    // 推理（禁用梯度计算，提升效率）
    torch::NoGradGuard no_grad;
    auto test_y_pred = model->forward(test_x);

    // 打印结果
    float true_y = std::sin(0.1f * 1003);
    std::cout << "输入序列：[" << test_seq[0] << ", " << test_seq[1] << ", " << test_seq[2] << "]" << std::endl;
    std::cout << "预测值：" << test_y_pred.item<float>() << std::endl;
    std::cout << "真实值：" << true_y << std::endl;
    std::cout << "误差：" << std::abs(test_y_pred.item<float>() - true_y) << std::endl;

 
}