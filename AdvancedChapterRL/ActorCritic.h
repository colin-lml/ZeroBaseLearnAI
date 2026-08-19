#pragma once

class ValueNetImpl : public torch::nn::Module
{
public:
	ValueNetImpl() = default;

	ValueNetImpl(int64_t input, int64_t output=1, int64_t hidden = 128)
	{
		m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
		m_fc2 = register_module("fc2", torch::nn::Linear(hidden, output));
	}


	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(m_fc1->forward(x));
		return m_fc2->forward(x);
	}

	torch::nn::Linear m_fc1{ nullptr };
	torch::nn::Linear m_fc2{ nullptr };

};

TORCH_MODULE(ValueNet);

class ActorCritic : public BaseAdvanced
{
public:
	

private:
	void GenerateTrainData(int maxCount) override;

	double TakeAction(VectorDouble& s0, bool bPredict = false) override;

	void TrainGenerateItem1(const QwItem& item) override {};
	void TrainGenerateItem2(const QwList& vList) override;
	

	PolicyNet m_ActorNet;
	ValueNet m_CriticNet;

	
	const double m_dbActorLR = 1e-3;
	const double m_dbCriticLR = 1e-2;

	torch::optim::Adam* m_pAdamActor;
	torch::optim::Adam* m_pAdamCritic;
	
};

