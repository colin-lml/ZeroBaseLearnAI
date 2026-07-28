#pragma once


class DuelingNetImpl : public torch::nn::Module
{
public:
	DuelingNetImpl() = default;
	DuelingNetImpl(int64_t input, int64_t output, int64_t hidden = 128)
	{
		m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
		m_A = register_module("A", torch::nn::Linear(hidden, output));
		m_V = register_module("V", torch::nn::Linear(hidden, 1));
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

};
TORCH_MODULE(DuelingNet);

class DuelingDQN : public  BaseAdvanced
{
public:

protected:

	void GenerateTrainData(int maxCount) override;
	int TakeAction(VectorDouble& s0, bool bPredict = false) override;

	void TrainGenerateItem1(const QwItem& item) override;
	void TrainGenerateItem2(const QwList& vList) override {};

	void CreateOptimizer(DuelingNet& model);
	void Update();

	void SyncTargetNet();

	XRandom m_xRandomData;
	DuelingNet m_Qnet;
	DuelingNet m_TargetQnet;

	const int m_nMinimalsize = 500;
	const int64_t m_batchsize = 64;
	torch::optim::Adam* m_pAdam = nullptr;
};

