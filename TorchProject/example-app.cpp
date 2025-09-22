#include <torch/torch.h>
#include <iostream>
#if 1

// ����ģ��
class ConvReluBnImpl : public torch::nn::Module {
public:
    ConvReluBnImpl(int input_channel = 3, int output_channel = 64, int kernel_size = 3, int stride = 1);// ���캯��������ģ���������
    torch::Tensor forward(torch::Tensor x);// ǰ�򴫲�
private:
    // ������Ҫʹ�õĲ����������
    torch::nn::Conv2d conv{ nullptr };// ������һ�������
    torch::nn::BatchNorm2d bn{ nullptr };// ����һ����һ����

};

TORCH_MODULE(ConvReluBn);// �������ģ��ȡ�����������ڵ���ģ��ʱ��ʹ���������������

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    return conv_options;// ����һ��Conv2dOptions����
}

// ���캯����ʹ�õĶ���������������ò���
ConvReluBnImpl::ConvReluBnImpl(int input_channel, int output_channel, int kernel_size, int stride) {
    // ����һ��conv_options
    conv = register_module("conv", torch::nn::Conv2d(conv_options(input_channel, output_channel, kernel_size, stride, kernel_size / 2)));
    // ֱ�����ò���
    bn = register_module("bn", torch::nn::BatchNorm2d(output_channel));

}
// 
torch::Tensor ConvReluBnImpl::forward(torch::Tensor x) {
    x = torch::relu(conv->forward(x));// ��ʽ����forward����
    x = bn(x);// ��ʽ����forward����������Ч����ͬ
    return x;
}

class CNN : public torch::nn::Module {
public:
    CNN(int in_channels, int out_channels);// ���캯��
    torch::Tensor forward(torch::Tensor x);// ǰ�򴫲�����
private:
    int mid_channels[3] = { 32,64,128 };
    // ����Ҫʹ�õ�ģ��
    ConvReluBn conv1{ nullptr };
    ConvReluBn down1{ nullptr };
    ConvReluBn conv2{ nullptr };
    ConvReluBn down2{ nullptr };
    ConvReluBn conv3{ nullptr };
    ConvReluBn down3{ nullptr };
    // ����Ҫʹ�õĲ�
    torch::nn::Conv2d out_conv{ nullptr };
};

CNN::CNN(int in_channels, int out_channels) {
    // ��ʼ��
    conv1 = ConvReluBn(in_channels, mid_channels[0], 3);
    down1 = ConvReluBn(mid_channels[0], mid_channels[0], 3, 2);
    conv2 = ConvReluBn(mid_channels[0], mid_channels[1], 3);
    down2 = ConvReluBn(mid_channels[1], mid_channels[1], 3, 2);
    conv3 = ConvReluBn(mid_channels[1], mid_channels[2], 3);
    down3 = ConvReluBn(mid_channels[2], mid_channels[2], 3, 2);
    out_conv = torch::nn::Conv2d(conv_options(mid_channels[2], out_channels, 3));
    // ������
    conv1 = register_module("conv1", conv1);
    down1 = register_module("down1", down1);
    conv2 = register_module("conv2", conv2);
    down2 = register_module("down2", down2);
    conv3 = register_module("conv3", conv3);
    down3 = register_module("down3", down3);
    out_conv = register_module("out_conv", out_conv);
}
torch::Tensor CNN::forward(torch::Tensor x) {
    // �������ݵ�����
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
    auto cnn_input = torch::randint(255, { 1,3,224,224 });// ����Ϊһ����������ͨ����224x224���������
    
    
    torch::optim::Adam optimizer_cnn(cnn.parameters(), 0.0003);// �Ż���������cnn����Ĳ�����ѧϰ��=0.003
    auto cnn_target = torch::zeros({ 1,1,26,26 });// Ŀ����Ϊһ������һ��ͨ����26x26��ȫ0����


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

  for (int i = 0; i < 30; i++) { // ѵ��50��
      optimizer_cnn.zero_grad();// �ݶ�����

      auto out = cnn.forward(cnn_input);// ���봫��
      auto loss = torch::mse_loss(out, cnn_target);// ������ʧ
      loss.backward();// ��ʧ���򴫵�

      optimizer_cnn.step();// ���²���

      std::cout << out[0][0][0] << std::endl;
  }

  std::cin.get();
}
#else



#include <torch/torch.h>

struct Net : torch::nn::Module {
    Net() {
        // �ڹ��캯���г�ʼ�������
        //conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 20, 5).stride(1).padding(2)));
       // conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 50, 5).stride(1).padding(2)));
        //fc1 = register_module("fc1", torch::nn::Linear(50 * 7 * 7, 500));
       // fc2 = register_module("fc2", torch::nn::Linear(500, 10));
        fc1 = register_module("fc1", torch::nn::Linear(20, 5));
    }

    // ʵ��ǰ�򴫲��߼�
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return x;
    }

    // ģ�Ͳ�
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