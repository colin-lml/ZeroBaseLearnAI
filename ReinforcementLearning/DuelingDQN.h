#pragma once


class VANentImpl : public torch::nn::Module
{
public:
	VANentImpl()
	{
		m_fc1 = register_module("fc1", torch::nn::Linear(inputN, hiddenN));
		m_A = register_module("A", torch::nn::Linear(hiddenN, outN));

		m_V = register_module("V", torch::nn::Linear(hiddenN, 1));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(m_fc1->forward(x));
		auto a  = m_A->forward(x);
		auto v = m_V->forward(x);

		return v + a - a.mean(1).view({-1,1});
	}

	torch::nn::Linear m_fc1{ nullptr };
	torch::nn::Linear m_V{ nullptr };
	torch::nn::Linear m_A{ nullptr };
	const int64_t inputN = 4;
	const int64_t hiddenN = 128;
	const int64_t outN = 2;
};
TORCH_MODULE(VANent);

class DuelingDQN
{
public:
	void PlayCartPole(int maxCount = 200);
private:

	void TestData();
	void TrainData(int maxCount);

	void SyncTargetNet();
	int TakeAction(VectorDouble s0);

	torch::optim::Adam CreateOptimizer(VANent& model);
	void TrainQnet(torch::optim::Adam& adam);

	CartPoleEnv m_CartPoleEnv;
	XRandom m_xRandomData;
	VANent m_Qnet;
	VANent m_TargetQnet;

	//const double m_dbAlpha = 0.1;
	const double m_dbGamma = 0.98;
	const double m_dbEpsilon = 0.01;
	const double m_dbLR = 1e-2;
	const int m_nMinimalsize = 500;
	const int64_t m_batchsize = 64;
	
};

