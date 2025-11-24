#include <stdlib.h> 
#include "torch/torch.h"
#include <chrono>
using namespace std;


struct CNNModule : torch::nn::Module
{
	CNNModule(int dim)
	{
		int s = 1;
		int p = 1;
		conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(torch::nn::Conv2dOptions(1, kernelOutchannel, Conv2kernel).stride(s).padding(p).bias(false)));
		max_pool2d = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(Pool2dkernel).stride(s));
		//conv2_drop = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.1));

		dim = (dim + 2 * p - (Conv2kernel - 1) - 1) / s + 1;
		p = 0;
		dim = (dim + 2 * p - (Pool2dkernel - 1) - 1) / s + 1;
		
		dimLinear = dim * dim * kernelOutchannel;
		
		in = register_module("in", torch::nn::Linear(dimLinear, 128));
		hide = register_module("hide", torch::nn::Linear(128, 64));
		out = register_module("out", torch::nn::Linear(64, 10));
	}

	torch::Tensor forward(torch::Tensor x)
	{
	 	x = conv->forward(x);
		//x = conv2_drop->forward(x);
		//x = torch::relu(x);
		x = max_pool2d->forward(x);
		pooldata = x;
		x = torch::relu(x);
		x = x.view({ -1,dimLinear });
		
		x = in->forward(x);
		x = torch::relu(x);

		x = hide->forward(x);
		x = torch::relu(x);
		
		x = out->forward(x);

		return torch::log_softmax(x, 1);
	}

	torch::nn::Linear in{ nullptr }, hide{ nullptr }, out{ nullptr };

	torch::nn::Conv2d conv{ nullptr };
	torch::nn::MaxPool2d max_pool2d{ nullptr };
	//torch::nn::Dropout2d conv2_drop;
	int64_t dimLinear = 0;
	int Conv2kernel = 3;
	int kernelOutchannel = 8;
	int Pool2dkernel = 2;
	torch::Tensor pooldata;
};

void TrainData(CNNModule& cnn) 
{
	auto dropout = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.5));

	std::cout << "train data .... " << endl << endl;
	cnn.train();

	torch::Tensor train1 = torch::tensor({ { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
										   { 0, 0, 0, 1, 1, 0, 0, 0,0,0},
		}, torch::kFloat).view({ 1, 1, 10, 10 });

	torch::Tensor train7 = torch::tensor({ { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											{ 0, 0, 0, 1, 1, 1, 1, 1,0,0},
											{ 0, 0, 0,  0, 0, 0, 0,1,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 1,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 1,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 1,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 1,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 0,0,0},
											{ 0, 0, 0, 0, 0, 0, 0, 0,0,0},
		}, torch::kFloat).view({ 1, 1, 10, 10 });

	torch::Tensor train8 = torch::tensor({ { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
										   { 0, 1, 1, 1, 1, 1, 1, 0,0,0},
										   { 0, 1, 0, 0, 0, 0, 1, 0,0,0},
										   { 0, 1, 0, 0, 0, 0, 1, 0,0,0},
										   { 0, 1, 1, 1, 1, 1, 1, 0,0,0},
										   { 0, 1, 1, 1, 1, 1, 1, 0,0,0},
										   { 0, 1, 0, 0, 0, 0, 1, 0,0,0},
										   { 0, 1, 0, 0, 0, 0, 1, 0,0,0},
										   { 0, 1, 1, 1, 1, 1, 1, 0,0,0},
										   { 0, 0, 0, 0, 0, 0, 0, 0,0,0},
		}, torch::kFloat).view({ 1, 1, 10, 10 });

	
	torch::Tensor target1 = torch::tensor({ 1 }, torch::kLong);
	torch::Tensor target7 = torch::tensor({ 7 }, torch::kLong);
	torch::Tensor target8 = torch::tensor({ 8 }, torch::kLong);

	double learning_rate = 0.01;

	torch::optim::Adam optimizer(cnn.parameters(), torch::optim::AdamOptions(learning_rate));


	int64_t epochs = 150;
	double accuracy = 0.008;
	auto start_time = chrono::high_resolution_clock::now();
	torch::Tensor pooldata1;
	torch::Tensor pooldata7;
	torch::Tensor pooldata8;

	for (int64_t epoch = 0; epoch < epochs; ++epoch)
	{
		optimizer.zero_grad();
		auto  output1 = cnn.forward(train1);
		auto loss1 = torch::nll_loss(output1, target1);
		loss1.backward();
		optimizer.step();
		pooldata1 = cnn.pooldata;

		auto output7 = cnn.forward(train7);
		auto loss7 = torch::nll_loss(output7, target7);
		loss7.backward();
		optimizer.step();
		pooldata7 = cnn.pooldata;

		auto  output8 = cnn.forward(train8);
		auto loss8 = torch::nll_loss(output8, target8);

		loss8.backward();
		optimizer.step();
		pooldata8 = cnn.pooldata;

		if (epoch % 10 == 0)
		{
			std::cout << " epoch: " << epoch + 1 << " , Loss0: " << loss8.item<double>() << ", loss1: " << loss1.item<double>() << ", loss7: " << loss7.item<double>() << std::endl;
		}

		if (loss8.item<double>() <= accuracy 
			&& loss1.item<double>() <= accuracy
			&& loss7.item<double>() <= accuracy)
		{
			std::cout << " epoch: " << epoch + 1 << " , Loss0: " << loss8.item<double>() << ", loss1: " << loss1.item<double>() << ", loss7: " << loss7.item<double>() << std::endl;
			
			break;
		}
	}

	
	//std::cout << endl << "pooldata1: " << endl << pooldata1 << endl;
	//std::cout << endl << "pooldata7: " << endl << pooldata7 << endl;
	//std::cout << endl << "pooldata8: " << endl << pooldata8 << endl;
	//std::cout << endl;


	auto end_time = chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	std::cout << "end-time: " << duration_ms << " ms, => " << duration_ms / 1000 << " s" << endl << endl;

}

pair<int, float>  MinLoss(int testid, torch::Tensor out1, torch::Tensor out7, torch::Tensor out8)
{
	auto target = torch::tensor({ testid }, torch::kLong);

	auto  loss1 = torch::nll_loss(out1, target).item<float>();
	auto  loss7 = torch::nll_loss(out7, target).item<float>();
	auto  loss8 = torch::nll_loss(out8, target).item<float>();
	
	int id = 1;
	float ltmp = loss1;
	if (loss7 < loss1)
	{
		id = 7;
		ltmp = loss7;
	}

	if (loss8 < ltmp)
	{
		id = 8;
		ltmp = loss8;
	}

	return { id,ltmp };

}

void TestData(CNNModule& cnn)
{
	torch::NoGradGuard no_grad;
	cnn.eval();
	std::cout << endl << "test data .... " << endl << endl;



	torch::Tensor test1 = torch::tensor({ { 0, 0, 0, 0,  1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
										   { 0, 0, 0, 0, 1, 1, 0, 0,0,0},
		}, torch::kFloat).view({ 1, 1, 10, 10 });

	torch::Tensor test7 = torch::tensor({   {  0, 0, 0, 0, 0, 0, 0,0,0,0},
										    {  0, 0, 0, 1, 1, 1, 1,0,0,0},
											{  0, 0,  0, 0, 0, 0,1,0,0,0},
											{  0, 0, 0, 0, 0,0,  1,0,0,0},
											{  0, 0, 0, 0, 0, 0, 1,0,0,0},
											{  0, 0, 0, 0, 0, 0, 1,0,0,0},
											{  0, 0, 0, 0, 0, 0, 1,0,0,0},
											{  0, 0, 0, 0, 0, 0, 0,0,0,0},
											{  0, 0, 0, 0, 0, 0, 0,0,0,0},
											{  0, 0, 0, 0, 0, 0, 0,0,0,0},
		}, torch::kFloat).view({ 1, 1, 10, 10 });

	torch::Tensor test8 = torch::tensor({ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
										   { 0, 0, 0, 1, 1, 1, 1, 1, 1, 0},
										   { 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
										   { 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
										   { 0, 0, 0, 1, 1, 1, 1, 1, 1, 0},
										   { 0, 0, 0, 1, 1, 1, 1, 1, 1, 0},
										   { 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
										   { 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
										   { 0, 0, 0, 1, 1, 1, 1, 1, 1, 0},
										   { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		}, torch::kFloat).view({ 1, 1, 10, 10 });


	auto  output1 = cnn.forward(test1);
	auto  output7 = cnn.forward(test7);
	auto  output8 = cnn.forward(test8);


	auto [idk1, idkloss1] = MinLoss(1, output1, output7, output8);
	auto [idk7, idkloss7] = MinLoss(7, output1, output7, output8);
	auto [idk8, idkloss8] = MinLoss(8, output1, output7, output8);

	
	for (int i = 0; i < 10; i++)
	{
		if (i==1 || i == 7 || i == 8)
		{
			auto target = torch::tensor({ i }, torch::kLong);
			auto  loss1 = torch::nll_loss(output1, target).item<float>();
			auto  loss7 = torch::nll_loss(output7, target).item<float>();
			auto  loss8 = torch::nll_loss(output8, target).item<float>();

			std::cout << "test data: " << i << " test-1-loss: " << loss1 << " , test-7-loss: " << loss7 << " , test-8-loss: " << loss8 << endl;

		}
	}
	std::cout  << endl;
	std::cout << "test data 1 - idk1: " << idk1 << " , loss: " << idkloss1<< endl;
	std::cout << "test data 7 - idk7: " << idk7 << " , loss: " << idkloss7<< endl;
	std::cout << "test data 8 - idk8: " << idk8 << " , loss: " << idkloss8<< endl;

}


void CnnMain()
{
	torch::manual_seed(1);

	CNNModule cnn(10);

	TrainData(cnn);

	TestData(cnn);

}