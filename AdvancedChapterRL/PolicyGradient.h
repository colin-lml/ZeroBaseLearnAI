#pragma once

class PolicyNetImpl : public torch::nn::Module
{
public:

	PolicyNetImpl() = default;

	PolicyNetImpl(int64_t input, int64_t output, int64_t hidden = 128)
	{
		m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
		m_fc2 = register_module("fc2", torch::nn::Linear(hidden, output));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(m_fc1->forward(x));
		x = m_fc2->forward(x);
		return torch::softmax(x, 1);
	}

	torch::nn::Linear m_fc1{ nullptr };
	torch::nn::Linear m_fc2{ nullptr };

};

TORCH_MODULE(PolicyNet);




class PolicyGradient : public BaseAdvanced
{
public:


protected:
	void GenerateTrainData(int maxCount) override;
	double TakeAction(VectorDouble& s0, bool bPredict = false) override;

	void TrainGenerateItem1(const QwItem& item) override {};
	void TrainGenerateItem2(const QwList& vList) override ;

	void CreateOptimizer(PolicyNet& model);

	PolicyNet m_Qnet;

	torch::optim::Adam* m_pAdam = nullptr;
};

