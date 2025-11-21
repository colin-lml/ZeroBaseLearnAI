#include <stdlib.h> 
#include "torch/torch.h"
#include <chrono>
using namespace std;


struct CNNModule : torch::nn::Module
{
	CNNModule(int dim)
	{
		conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(torch::nn::Conv2dOptions(1, kernelOutchannel, Conv2kernel).stride(1).padding(1)));
		max_pool2d = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(Pool2dkernel).stride(1));
		
		dim = dim + 3 - Conv2kernel;
		
		dimLinear = dim - Pool2dkernel + 1;
		dimLinear = dimLinear * dimLinear * kernelOutchannel;
		
		in = register_module("in", torch::nn::Linear(dimLinear, 512));
		hide = register_module("hide", torch::nn::Linear(512, 128));
		out = register_module("out", torch::nn::Linear(128, 10));

		in->to(torch::kFloat);
		hide->to(torch::kFloat);
		out->to(torch::kFloat);

	}

	torch::Tensor forward(torch::Tensor x)
	{
	 	x = conv->forward(x);
		x = max_pool2d->forward(x);
		x = torch::relu(x);
		x = x.view({ -1,dimLinear });
		
		x = in->forward(x);
		x = torch::relu(x);

		x = hide->forward(x);
		x = torch::relu(x);

		x = out->forward(x);

		return torch::log_softmax(x, 1);
	}

	torch::nn::Linear in{ nullptr }, hide{ nullptr },out{ nullptr };

	torch::nn::Conv2d conv{ nullptr };
	torch::nn::MaxPool2d max_pool2d{ nullptr };
	int64_t dimLinear = 0;
	int Conv2kernel = 3;
	int kernelOutchannel = 16;
	int Pool2dkernel = 3;

};





void CnnMain()
{
	torch::manual_seed(1);

	torch::Tensor input1 = torch::tensor({{ 0, 0, 0, 0, 0, 0, 0, 0,0,0},
										  { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										  { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										  { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										  { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										  { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										  { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										  { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
										  { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
										  { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
		}, torch::kFloat).view({ 1, 1, 10, 10 });
		
	  torch::Tensor input7 = torch::tensor({{ 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											{ 0, 1, 1, 1, 1, 1, 0, 0,0,0},
											{ 0, 1, 1, 1, 1, 1, 0, 0,0,0},
										    { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										    { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										    { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										    { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										    { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 0,0,0},
		}, torch::kFloat).view({ 1, 1, 10, 10 });

	 torch::Tensor target1 = torch::tensor({ 1 }, torch::kLong);
	 torch::Tensor target7 = torch::tensor({ 7 }, torch::kLong);

	CNNModule cnn(input1.size(2));

	double learning_rate = 0.01;

	torch::optim::Adam optimizer(cnn.parameters(), torch::optim::AdamOptions(learning_rate));
	
	
	int64_t epochs = 10000;
	double accuracy = 0.003;
	auto start_time = chrono::high_resolution_clock::now();
	
	cnn.train();
	
	for (int64_t epoch = 0; epoch < epochs; ++epoch)
	{
		optimizer.zero_grad();

		auto  output = cnn.forward(input1);
		auto loss = torch::nll_loss(output, target1);

		loss.backward();
		optimizer.step();

		auto output7 = cnn.forward(input7);
		auto loss7 = torch::nll_loss(output7, target7);
		loss7.backward();
		optimizer.step();
		if (epoch % 10 == 0)
		{
			std::cout << "Epoch: " << epoch + 1 << "], Loss1: " << loss << ", loss7:" << loss7 << std::endl;
		}
			
		if (loss.item<double>() <= accuracy && loss7.item<double>() <= accuracy) 
		{
			std::cout << "Epoch: " << epoch + 1 << "], Loss1: " << loss << ", loss7:" << loss7 << std::endl;
			break;
		}
	}
	
	auto end_time = chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	std::cout << "end-time: " << duration_ms <<" ms, => "<< duration_ms/1000 <<" s" << endl;
	
	{

		torch::Tensor test1 = torch::tensor({ { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											  { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											  { 0, 0, 0, 0, 0, 1, 1, 0,0,0},
											  { 0, 0, 0, 0, 0, 1, 1, 0,0,0},
											  { 0, 0, 0, 0, 0, 1, 1, 0,0,0},
											  { 0, 0, 0, 0, 0, 1, 1, 0,0,0},
											  { 0, 0, 0, 0, 0, 1, 1, 0,0,0},
											  { 0, 0, 0, 0, 0, 1, 1, 0,0,0},
											  { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											  { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
			}, torch::kFloat).view({ 1, 1, 10, 10 });


		torch::NoGradGuard no_grad;
		cnn.eval();
		auto  output = cnn.forward(test1);
		auto  loss = torch::nll_loss(output, target1);
		std::cout << endl << "eval-output:\n "  << endl << "loss: " << loss << endl;

	}


}