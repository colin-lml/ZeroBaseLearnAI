#pragma once

class ValueNetImpl : public torch::nn::Module
{
public:
	ValueNetImpl()
	{
		m_fc1 = register_module("fc1", torch::nn::Linear(inputN, hiddenN));
		m_fc2 = register_module("fc2", torch::nn::Linear(hiddenN, outN));
	}


	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(m_fc1->forward(x));
		return m_fc2->forward(x);
	}

	torch::nn::Linear m_fc1{ nullptr };
	torch::nn::Linear m_fc2{ nullptr };

	const int64_t inputN = 4;
	const int64_t hiddenN = 128;
	const int64_t outN = 1;
};

TORCH_MODULE(ValueNet);

class ActorCritic
{
public:
	void PlayCartPole(int maxCount = 500);

private:

	int TakeAction(VectorDouble s0, bool bPredict = false);
	void TestData();
	void TrainData(int maxCount);
	void Update(torch::optim::Adam& a, torch::optim::Adam& c, QwList& vList);

	PolicyNet m_ActorNet;
	ValueNet m_CriticNet;

	const double m_dbGamma = 0.98;
	//const double m_dbEpsilon = 0.1;
	const double m_dbActorLR = 1e-3;
	const double m_dbCriticLR = 1e-2;

	CartPoleEnv m_CartPoleEnv;
};

