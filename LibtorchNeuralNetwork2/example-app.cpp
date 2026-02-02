// LibtorchSimpleNeuralNetwork.cpp: 定义应用程序的入口点。
//

#include <stdlib.h> 
#include "torch/torch.h"
#include <chrono>


using namespace std;
void CnnMain();
void RnnMain();
int autogradMain();
void EmbeddingMain();
void TransformerMain();

void ResNetMain();

struct NetModule : torch::nn::Module
{
	NetModule()
	{
		fc1 = register_module("fc1", torch::nn::Linear(2, 2));
		fc2 = register_module("fc2", torch::nn::Linear(2, 2));
		fc1->to(torch::kDouble);
		fc2->to(torch::kDouble);


		fc1->weight.set_data(torch::tensor({ {0.15,0.20},
											 {0.25, 0.30} }, torch::kDouble));
		fc1->bias.set_data(torch::tensor({ 0.35,0.35 }, torch::kDouble));


		fc2->weight.set_data(torch::tensor({ {0.40,0.45}
										   ,{0.50,0.55} }, torch::kDouble));
		fc2->bias.set_data(torch::tensor({ 0.60,0.60 }, torch::kDouble));


		cout << "<<<-------------------------------------------------->>>" << endl;
		std::cout << std::fixed << std::setprecision(10);
		cout << fc1->weight << endl << fc1->bias << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << fc2->weight << endl << fc2->bias << endl;
		cout << "<<<<-------------------------------------------------- >>>" << endl << endl;

	}

	torch::Tensor forward(torch::Tensor x)
	{
		auto t = torch::sigmoid(fc1->forward(x));
		auto l2 = torch::sigmoid(fc2->forward(t));

		return 	l2;
	}



	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

};

struct ResNetModule : torch::nn::Module 
{

	ResNetModule(double w)
	{
		torch::nn::LinearOptions linear(1,1);
		linear.bias(false);
		fc1 = register_module("fc1", torch::nn::Linear(linear));
		fc2 = register_module("fc2", torch::nn::Linear(linear));
		fc3 = register_module("fc3", torch::nn::Linear(linear));
		fc4 = register_module("fc4", torch::nn::Linear(linear));
		fc5 = register_module("fc5", torch::nn::Linear(linear));
		fc6 = register_module("fc6", torch::nn::Linear(linear));
		fc7 = register_module("fc7", torch::nn::Linear(linear));
		fc8 = register_module("fc8", torch::nn::Linear(linear));
		fc9 = register_module("fc9", torch::nn::Linear(linear));
		fc10 = register_module("fc10", torch::nn::Linear(linear));

		fc1->to(torch::kDouble);
		fc2->to(torch::kDouble);
		fc3->to(torch::kDouble);
		fc4->to(torch::kDouble);
		fc5->to(torch::kDouble);
		fc6->to(torch::kDouble);
		fc7->to(torch::kDouble);
		fc8->to(torch::kDouble);
		fc9->to(torch::kDouble);
		fc10->to(torch::kDouble);

		/* x
		fc1->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc2->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc3->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc4->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc5->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc6->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc7->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc8->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc9->weight.set_data(torch::tensor({ w }, torch::kDouble));
		fc10->weight.set_data(torch::tensor({ w }, torch::kDouble));
		*/
		
	}
	torch::Tensor NormalForward(torch::Tensor x)
	{
		auto t = torch::relu(fc1->forward(x));
		t = torch::relu(fc2->forward(t.view(1)));
		t = torch::relu(fc3->forward(t.view(1)));
		t = torch::relu(fc4->forward(t.view(1)));
		t = torch::relu(fc5->forward(t.view(1)));
		t = torch::relu(fc6->forward(t.view(1)));
		t = torch::relu(fc7->forward(t.view(1)));
		t = torch::relu(fc8->forward(t.view(1)));
		t = torch::relu(fc9->forward(t.view(1)));
		t = torch::relu(fc10->forward(t.view(1)));
		return t.view(1);
	}



	torch::Tensor ResnetForward(torch::Tensor x)
	{
		auto t = torch::relu(fc1->forward(x)+x);
		t = torch::relu(fc2->forward(t.view(1)) + t.view(1));
		t = torch::relu(fc3->forward(t.view(1)) + t.view(1));
		t = torch::relu(fc4->forward(t.view(1)) + t.view(1));
		t = torch::relu(fc5->forward(t.view(1)) + t.view(1));
		t = torch::relu(fc6->forward(t.view(1)) + t.view(1));
		t = torch::relu(fc7->forward(t.view(1)) + t.view(1));
		t = torch::relu(fc8->forward(t.view(1)) + t.view(1));
		t = torch::relu(fc9->forward(t.view(1)) + t.view(1));
		t = torch::relu(fc10->forward(t.view(1)) + t.view(1));
		return t.view(1);
	}
	torch::Tensor ResnetForward2(torch::Tensor x)
	{
		auto t = torch::relu(fc1->forward(x));
		t = torch::relu(fc2->forward(t.view(1)));
		t = torch::relu(fc3->forward(t.view(1)));
		t = torch::relu(fc4->forward(t.view(1)));
		t = torch::relu(fc5->forward(t.view(1)));
		t = torch::relu(fc6->forward(t.view(1)));
		t = torch::relu(fc7->forward(t.view(1)));
		t = torch::relu(fc8->forward(t.view(1)));
		t = torch::relu(fc9->forward(t.view(1)));
		t = torch::relu(fc10->forward(t.view(1)) + x);
		return t.view(1);
	}

	void ShowWeigth()
	{
		std::cout << endl<<"权重:"<< std::fixed << std::setprecision(10)<< endl;
		std::cout << fc1->weight  << endl;
		std::cout << fc2->weight << endl;
		std::cout << fc3->weight << endl;
		std::cout << fc4->weight << endl;
		std::cout << fc5->weight << endl;
		std::cout << fc6->weight << endl;
		std::cout << fc7->weight << endl;
		std::cout << fc8->weight << endl;
		std::cout << fc9->weight << endl;
		std::cout << fc10->weight << endl;
	}

	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr }, fc5{ nullptr }
	, fc6{ nullptr }, fc7{ nullptr }, fc8{ nullptr }, fc9{ nullptr }, fc10{ nullptr };
};



int main()
{
	double learning_rate = 0.03;   ///学习率
	int64_t epochs = 1000;         /// 训练最大次数
	int64_t i = 0;
	double accuracy = 0.0000006;  // 与目标值比较精度
	ResNetModule minloss(2.05);//初始化权重值
	torch::nn::MSELoss funloss;
	torch::optim::Adam optimizer(minloss.parameters(), torch::optim::AdamOptions(learning_rate));
	auto input = torch::tensor({ 1.0 }, torch::kDouble);
	auto labels = torch::tensor({ 1.0 }, torch::kDouble);  // 目标值
	double y = 0;
	
	for (i = 0; i < epochs; ++i)
	{
		//auto out =  minloss.NormalForward(input); // 传统神经网络
		//auto out = minloss.ResnetForward(input); //残差网络1
		auto out = minloss.ResnetForward2(input);//残差网络2
		auto loss = funloss(out, labels);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		y = out.index({ 0 }).item<double>();

		if (abs(y - labels.index({ 0 }).item<double>()) <= accuracy)
		{
			break;
		}
	}

	std::cout<<"训练次数: " << i <<",  目标值: " <<y << std::endl;
	minloss.ShowWeigth();

	//autogradMain();
	//CnnMain();
	//RnnMain();

	//EmbeddingMain();

	//TransformerMain();
	// ResNetMain();
#if 0
	NetModule net;
	double learning_rate = 0.5;
	//torch::Device device(torch::kCPU);
	//net.to(device);
	torch::nn::MSELoss funloss;
	torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(learning_rate));
	auto input = torch::tensor({ 0.050,0.10 }, torch::kDouble);
	auto labels = torch::tensor({ 0.10,0.99 }, torch::kDouble);

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

		if (abs(out.index({ 0 }).item<double>() - labels.index({ 0 }).item<double>()) <= accuracy &&
			abs(out.index({ 1 }).item<double>() - labels.index({ 1 }).item<double>()) <= accuracy)
		{
			std::cout << " break [" << epoch + 1 << "/" << epochs << "], Loss: " << loss.item<double>() << ", out " << out << std::endl;
			break;
		}


		if ((epoch + 1) % 100 == 0)
		{
			std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Loss: " << loss << std::endl;
		}

	}

	auto end_time = chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	std::cout << "end-time " << duration_ms << endl;


	cout << "<<<-------------------------------------------------->>>" << endl;
	cout << net.fc1->weight << endl << net.fc1->bias << endl;
	cout << "-----------------------------------------------------" << endl;
	cout << net.fc2->weight << endl << net.fc2->bias << endl;
	cout << "<<<<-------------------------------------------------- >>>" << endl << endl;

	
	
#endif

	cin.get();
	return 0;
}

/*
*

Epoch [100/300000], Loss: 0.004742825631
[ CPUDoubleType{} ]
Epoch [200/300000], Loss: 7.291057732e-07
[ CPUDoubleType{} ]
Epoch [300/300000], Loss: 2.949105029e-10
[ CPUDoubleType{} ]
 break [393/300000], Loss: 0.0000000000, out  0.1000   0.9900
[ CPUDoubleType{2} ]

end-time 2943
*/
