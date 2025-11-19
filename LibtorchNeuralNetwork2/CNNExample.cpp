#include <stdlib.h> 
#include "torch/torch.h"
#include <chrono>
using namespace std;


void CnnMain()
{

	torch::Tensor input = torch::tensor({ { 1.0, 2.0, 3.0, 4.0, 5.0  },
										  { 1.0, 2.0, 3.0, 4.0, 5.0  },
										  { 1.0, 2.0, 3.0, 4.0, 5.0  },
										  { 1.0, 2.0, 3.0, 4.0, 5.0  },
										  { 1.0, 2.0, 3.0, 4.0, 5.0  },
		}).view({ 1, 1, 5, 5 });

	torch::Tensor input1 = torch::tensor({{ 0, 0, 0, 0, 0, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 1, 1, 0, 0, 0},
										  { 0, 0, 0, 0, 0, 0, 0, 0},
		}, torch::kDouble).view({ 1, 1, 8, 8 });
		
	  torch::Tensor input7 = torch::tensor({{ 0, 0, 0, 0, 0, 0, 0, 0},
											{ 0, 1, 1, 1, 1, 1, 0, 0},
											{ 0, 1, 1, 1, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 1, 1, 0, 0},
										    { 0, 0, 0, 0, 0, 0, 0, 0},
		}, torch::kDouble).view({ 1, 1, 8, 8 });

	  std::cout << "输出1: \n" << input1 << std::endl;
	  std::cout << "输出2: \n" << input7 << std::endl;

	
	// 步长1，用一个填充位
	torch::nn::Conv2d conv(torch::nn::Conv2dOptions(1, 3, 3).stride(1).padding(0));

	//conv->weight.data() = torch::tensor({ { 1.0, 0.0, 1.0 }, { 1.0, 0.0, 1.0 }, { 1.0, 0.0, 1.0 } });
	std::cout << "卷积核: \n" << conv->weight << std::endl;


	torch::Tensor output = conv->forward(input);   //卷积运算
	std::cout << "卷积输出结果: \n" << output << std::endl;

	//最大池化 窗口大小 3X3 步长1 
	//torch::nn::MaxPool2d max_pool2d(torch::nn::MaxPool2dOptions(3).stride(1));

	//torch::Tensor output2 = max_pool2d->forward(output);

	// 4. 输出结果
	///std::cout << "池化输出结果:\n" << output2 << std::endl;
}