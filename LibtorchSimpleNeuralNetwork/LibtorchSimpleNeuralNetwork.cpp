﻿// LibtorchSimpleNeuralNetwork.cpp: 定义应用程序的入口点。
//

#include <stdlib.h> 
#include "torch/torch.h"
#include <chrono>

#include "LibtorchSimpleNeuralNetwork.h"
using namespace std;



struct NetModule : torch::nn::Module
{
	NetModule()
	{
		fc1 = register_module("fc1", torch::nn::Linear(2, 2));
		fc2 = register_module("fc2", torch::nn::Linear(2, 2));
		fc1->to(torch::kDouble);
		fc2->to(torch::kDouble);
	
		fc1->weight.set_data(torch::tensor({ {0.15,0.25},
											 {0.20, 0.30} }, torch::kDouble));
		fc1->bias.set_data(torch::tensor({ 0.35,0.35 }, torch::kDouble));


		fc2->weight.set_data(torch::tensor({ {0.40,0.50}
										   ,{0.45,0.55} }, torch::kDouble));
		fc2->bias.set_data(torch::tensor({ 0.60,0.60 }, torch::kDouble));

	
		cout << "<<<-------------------------------------------------->>>" << endl;
		std::cout << std::fixed << std::setprecision(10);
		cout << fc1->weight << endl << fc1->bias << endl;
		cout<<"-----------------------------------------------------" << endl;
		cout << fc2->weight << endl << fc2->bias << endl;
		cout << "<<<<-------------------------------------------------- >>>" << endl << endl;

	}
	
	torch::Tensor forward(torch::Tensor x) 
	{	
		auto t = torch::sigmoid(fc1->forward(x));
		auto l2 = torch::sigmoid(fc2->forward(t));
		//cout << "---------- >>> " << x << endl << endl;
		return 	l2;
	}

	

	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

};





int main()
{

	NetModule net;
	float learning_rate = 0.5;
	//torch::Device device(torch::kCPU);
	///net.to(device);
	torch::nn::MSELoss funloss;
	torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(learning_rate));
	auto input = torch::tensor({ {0.050,0.10} }, torch::kDouble);
	auto labels = torch::tensor({ {0.10,0.99} }, torch::kDouble);

	int64_t epochs = 10000 * 30;

	double accuracy = 0.0000006;
	
	auto start_time = chrono::high_resolution_clock::now();

	for (int64_t epoch = 0; epoch < epochs; ++epoch)
	{
		auto out = net.forward(input);

		auto loss = funloss(out, labels);

		optimizer.zero_grad();  
		loss.backward();        
		optimizer.step();   
		
		if (abs(out.index({ 0,0 }).item<double>() - labels.index({ 0,0 }).item<double>()) <= accuracy &&
			abs(out.index({ 0,1 }).item<double>() - labels.index({ 0,1 }).item<double>()) <= accuracy)
		{
			std::cout << " break [" << epoch + 1 << "/" << epochs << "], Loss: " << loss.item<double>() <<", out "<< out << std::endl;
			break;
		}

		
		if ((epoch + 1) % 100 == 0) 
		{
			std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Loss: " << loss << std::endl;
		}
		
	}

	auto end_time = chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	std::cout << "end-time " << duration_ms<< endl;


	cout << "<<<-------------------------------------------------->>>" << endl;
	cout << net.fc1->weight << endl << net.fc1->bias << endl;
	cout << "-----------------------------------------------------" << endl;
	cout << net.fc2->weight << endl << net.fc2->bias << endl;
	cout << "<<<<-------------------------------------------------- >>>" << endl << endl;
	
	cin.get();

	return 0;
}

/*
* 
 Epoch [100/300000], Loss: 0.0031318587716668844
[ CPUFloatType{} ]
Epoch [200/300000], Loss: 1.4559006444869738e-07
[ CPUFloatType{} ]
Epoch [300/300000], Loss: 1.6511814138198133e-10
[ CPUFloatType{} ]
 break [394/300000], Loss: 1.44329e-13, out  0.1000  0.9900
[ CPUFloatType{1,2} ]
end-time 1479
*/
