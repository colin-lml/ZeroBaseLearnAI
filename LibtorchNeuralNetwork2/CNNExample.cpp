#include <stdlib.h> 
#include "torch/torch.h"
#include <chrono>
using namespace std;


struct CNNModule : torch::nn::Module
{
	CNNModule()
	{
		conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(torch::nn::Conv2dOptions(1, 3, 3).stride(1).padding(0)));
		max_pool2d = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(1));
		
		in = register_module("in", torch::nn::Linear(75, 100));
		out = register_module("out", torch::nn::Linear(100, 2));

		in->to(torch::kFloat);
		out->to(torch::kFloat);

	}

	torch::Tensor forward(torch::Tensor x)
	{
	 	x = conv->forward(x);
		x = max_pool2d->forward(x);
		x = torch::relu(x);
		x = x.view(-1);
		///std::cout << "卷积层输出结果: \n" << x << std::endl;

		x = in->forward(x);
		x = torch::relu(x);
		x = out->forward(x);
		std::cout << "输出结果xx: \n" << x << std::endl;
		return torch::log_softmax(x, /*dim=*/1);
	}



	torch::nn::Linear in{ nullptr }, out{ nullptr };

	torch::nn::Conv2d conv{ nullptr };
	torch::nn::MaxPool2d max_pool2d{ nullptr };
};





void CnnMain()
{

	torch::Tensor input1 = torch::tensor({{ 0, 0, 0, 0, 0, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 0, 0, 0, 0, 0},
		}, torch::kFloat).view({ 1, 1, 8, 8 });
		
	  torch::Tensor input7 = torch::tensor({{ 0, 0, 0, 0, 0, 0, 0, 0},
											{ 0, 1, 1, 1, 1, 1, 0, 0},
											{ 0, 1, 1, 1, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 0, 0, 0, 0},
		}, torch::kFloat).view({ 1, 1, 8, 8 });


#if 0	
	// 步长1，用一个填充位
	torch::nn::Conv2d conv(torch::nn::Conv2dOptions(1, 3, 3).stride(1).padding(0));

	//conv->weight.data() = torch::tensor({ { 1.0, 0.0, 1.0 }, { 1.0, 0.0, 1.0 }, { 1.0, 0.0, 1.0 } });
	std::cout << "卷积核: \n" << conv->weight << std::endl;


	torch::Tensor output = conv->forward(input1);   //卷积运算
	std::cout << "卷积输出结果: \n" << output << std::endl;

	//最大池化 窗口大小 2X2 步长1 
	torch::nn::MaxPool2d max_pool2d(torch::nn::MaxPool2dOptions(2).stride(1));

	torch::Tensor output2 = max_pool2d->forward(output); 
	torch::Tensor output3 = output2.view({ -1});
	
	std::cout << "池化输出结果:\n" << output2 << std::endl;
	std::cout << "output2.view({ -1}):\n" << output3 << std::endl;
#endif


	CNNModule cnn;
	double learning_rate = 0.5;

	torch::nn::MSELoss funloss;
	torch::optim::Adam optimizer(cnn.parameters(), torch::optim::AdamOptions(learning_rate));

	torch::Tensor  output= cnn.forward(input1);
	std::cout << "output:\n" << output << std::endl;
}