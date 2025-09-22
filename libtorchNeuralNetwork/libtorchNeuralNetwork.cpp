// libtorchNeuralNetwork.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>

#include "torch/torch.h"

class ConvReluBnImpl : public torch::nn::Module {
public:
    ConvReluBnImpl(int input_channel = 3, int output_channel = 64, int kernel_size = 3, int stride = 1);// 构造函数，传入模块所需参数
    torch::Tensor forward(torch::Tensor& x);// 前向传播
private:
    // 声明需要使用的层或其他功能
    torch::nn::Conv2d conv{ nullptr };// 声明了一个卷积层
    torch::nn::BatchNorm2d bn{ nullptr };// 声明一个归一化层
};
TORCH_MODULE(ConvReluBn);// 给上面的模块取别名，网络在调用模块时，使用这个名称来创建



class CNN : public torch::nn::Module {
public:
    CNN(int in_channels, int out_channels);// 构造函数
    torch::Tensor forward(torch::Tensor& x);// 前向传播函数
private:
    int mid_channels[3] = { 32,64,128 };
    // 声明要使用的模块
    ConvReluBn conv1{ nullptr };
    ConvReluBn down1{ nullptr };
    ConvReluBn conv2{ nullptr };
    ConvReluBn down2{ nullptr };
    ConvReluBn conv3{ nullptr };
    ConvReluBn down3{ nullptr };
    // 声明要使用的层
    torch::nn::Conv2d out_conv{ nullptr };
};

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    return conv_options;// 返回一个Conv2dOptions对象
}

// 构造函数对使用的对象进行命名与设置参数
ConvReluBnImpl::ConvReluBnImpl(int input_channel, int output_channel, int kernel_size, int stride) {
    // 传入一个conv_options
    conv = register_module("conv", torch::nn::Conv2d(conv_options(input_channel, output_channel, kernel_size, stride, kernel_size / 2)));
    // 直接设置参数
    bn = register_module("bn", torch::nn::BatchNorm2d(output_channel));

}
// 
torch::Tensor ConvReluBnImpl::forward(torch::Tensor& x) {
    x = torch::relu(conv->forward(x));// 显式调用forward函数
    x = bn(x);// 隐式调用forward函数，二者效果相同
    return x;
}




CNN::CNN(int in_channels, int out_channels) {
    // 初始化
    conv1 = ConvReluBn(in_channels, mid_channels[0], 3);
    down1 = ConvReluBn(mid_channels[0], mid_channels[0], 3, 2);
    conv2 = ConvReluBn(mid_channels[0], mid_channels[1], 3);
    down2 = ConvReluBn(mid_channels[1], mid_channels[1], 3, 2);
    conv3 = ConvReluBn(mid_channels[1], mid_channels[2], 3);
    down3 = ConvReluBn(mid_channels[2], mid_channels[2], 3, 2);
    out_conv = torch::nn::Conv2d(conv_options(mid_channels[2], out_channels, 3));
    // 重命名
    conv1 = register_module("conv1", conv1);
    down1 = register_module("down1", down1);
    conv2 = register_module("conv2", conv2);
    down2 = register_module("down2", down2);
    conv3 = register_module("conv3", conv3);
    down3 = register_module("down3", down3);
    out_conv = register_module("out_conv", out_conv);
}
torch::Tensor CNN::forward(torch::Tensor& x) {
    // 控制数据的流动
    x = conv1->forward(x);
    x = down1->forward(x);
    x = conv2->forward(x);
    x = down2->forward(x);
    x = conv3->forward(x);
    x = down3->forward(x);
    x = out_conv->forward(x);
    return x;
}



int main() 
{
#if 0 
    auto cnn = CNN(3, 1);
    auto cnn_input = torch::randint(255, { 1,3,224,224 });// 输入为一个包含三个通道的224x224的随机张量
    torch::optim::Adam optimizer_cnn(cnn.parameters(), 0.3);// 优化器，传入cnn网络的参数，学习率=0.003
    auto cnn_target = torch::zeros({ 1,1,26,26 });// 目标结果为一个包含一个通道的26x26的全0张量

    for (int i = 0; i < 30; i++) { // 训练50次
        //optimizer_cnn.zero_grad();// 梯度清零

       // cout << cnn_input<<endl;

        auto out = cnn.forward(cnn_input);// 输入传递
        auto loss = torch::mse_loss(out, cnn_target);// 计算损失
        loss.backward();// 损失反向传递

        optimizer_cnn.step();// 更新参数

        cout << out[0][0][0];
    }
#endif



}
