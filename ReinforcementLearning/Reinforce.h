#pragma once

class PolicyNetImpl : public torch::nn::Module
{
public:
	PolicyNetImpl()
	{
		m_fc1 = register_module("fc1", torch::nn::Linear(inputN, hiddenN));
		m_fc2 = register_module("fc2", torch::nn::Linear(hiddenN, outN));
	}


	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(m_fc1->forward(x));
		x = m_fc2->forward(x);
		return x.softmax(1);
	}

	torch::nn::Linear m_fc1{ nullptr };
	torch::nn::Linear m_fc2{ nullptr };

	const int64_t inputN = 4;
	const int64_t hiddenN = 128;
	const int64_t outN = 2;
};
TORCH_MODULE(PolicyNet);


class Reinforce
{
public:
	void PlayCartPole(int maxCount = 500);

private:
	int TakeAction(VectorDouble s0);
	void TestData();
	void TrainData(int maxCount);

	torch::optim::Adam CreateOptimizer(PolicyNet& model);


	PolicyNet m_Qnet;
	const double m_dbGamma = 0.98;
	//const double m_dbEpsilon = 0.1;
	const double m_dbLR = 1e-3;

	CartPoleEnv m_CartPoleEnv;
};

