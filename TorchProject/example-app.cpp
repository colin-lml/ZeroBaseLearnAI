#include <torch/torch.h>
#include <iostream>
#if 1

// 声明模块
class ConvReluBnImpl : public torch::nn::Module {
public:
    ConvReluBnImpl(int input_channel = 3, int output_channel = 64, int kernel_size = 3, int stride = 1);// 构造函数，传入模块所需参数
    torch::Tensor forward(torch::Tensor x);// 前向传播
private:
    // 声明需要使用的层或其他功能
    torch::nn::Conv2d conv{ nullptr };// 声明了一个卷积层
    torch::nn::BatchNorm2d bn{ nullptr };// 声明一个归一化层

};

TORCH_MODULE(ConvReluBn);// 给上面的模块取别名，网络在调用模块时，使用这个名称来创建

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
torch::Tensor ConvReluBnImpl::forward(torch::Tensor x) {
    x = torch::relu(conv->forward(x));// 显式调用forward函数
    x = bn(x);// 隐式调用forward函数，二者效果相同
    return x;
}

class CNN : public torch::nn::Module {
public:
    CNN(int in_channels, int out_channels);// 构造函数
    torch::Tensor forward(torch::Tensor x);// 前向传播函数
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
torch::Tensor CNN::forward(torch::Tensor x) {
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

    auto cnn = CNN(3, 1);
    auto cnn_input = torch::randint(255, { 1,3,224,224 });// 输入为一个包含三个通道的224x224的随机张量
    
    
    torch::optim::Adam optimizer_cnn(cnn.parameters(), 0.0003);// 优化器，传入cnn网络的参数，学习率=0.003
    auto cnn_target = torch::zeros({ 1,1,26,26 });// 目标结果为一个包含一个通道的26x26的全0张量


  torch::Tensor tensor = torch::rand({2, 3});
  if (torch::cuda::is_available())
  {
	  std::cout << "CUDA is available! Training on GPU" << std::endl;
	  //auto tensor_cuda = tensor.cuda();
	  //std::cout << tensor_cuda << std::endl;
  }
  else
  {
	 // std::cout << "CUDA is not available! Training on CPU" << std::endl;
	  //std::cout << tensor << std::endl;	  
  }

  for (int i = 0; i < 30; i++) { // 训练50次
      optimizer_cnn.zero_grad();// 梯度清零

      auto out = cnn.forward(cnn_input);// 输入传递
      auto loss = torch::mse_loss(out, cnn_target);// 计算损失
      loss.backward();// 损失反向传递

      optimizer_cnn.step();// 更新参数

      std::cout << out[0][0][0] << std::endl;
  }

  std::cin.get();
}
#else



#include <torch/torch.h>

struct Net : torch::nn::Module {
    Net() {
        // 在构造函数中初始化网络层
        //conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 20, 5).stride(1).padding(2)));
       // conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 50, 5).stride(1).padding(2)));
        //fc1 = register_module("fc1", torch::nn::Linear(50 * 7 * 7, 500));
       // fc2 = register_module("fc2", torch::nn::Linear(500, 10));
        fc1 = register_module("fc1", torch::nn::Linear(20, 5));
    }

    // 实现前向传播逻辑
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return x;
    }

    // 模型层
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

int main() {
    Net model;
    auto x = torch::randn({20});
    std::cout << x << std::endl;
    auto y = model.forward(x);
    std::cout << y << std::endl;
    std::cin.get();
    return 0;
}
#endif