#pragma once

QwList& GetCartPoleDataList();
void AddCartPoleDataList(const QwItem& item);


class ReplayBuffer
{
public:

	size_t size() const
	{
		return GetCartPoleDataList().size();
	}

	QwList sample(int batchsize)
	{
		QwList2D batch;
		QwList output;
		XRandom random;

		int count = size();
		count = min(count, batchsize);

		output.reserve(count);

		auto& datas = GetCartPoleDataList();
		std::sample(datas.begin(), datas.end(), std::back_inserter(output), count, random.GetGen());
	
		return output;
	}


private:
	
};


class DQNQnetImpl : public torch::nn::Module
{
public:
	DQNQnetImpl() = default;
	DQNQnetImpl(int64_t input, int64_t output, int64_t hidden=128)
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

TORCH_MODULE(DQNQnet);



class DeepQNetwork :public  BaseAdvanced
{
public:
	DeepQNetwork() = default;
	
protected:

	void GenerateTrainData(int maxCount) override;
	int TakeAction(VectorDouble s0, bool bPredict = false) override;
	void TrainGenerateItem1(const QwItem& item) override;
	void TrainGenerateItem2(const QwList& item) override {};

	void Update();

	void SyncTargetNet();
	void CreateOptimizer(DQNQnet& model);

	DQNQnet m_Qnet;
	DQNQnet m_TargetQnet;
	CartPoleEnv m_CartPoleEnv;

	const int m_nMinimalsize = 500;
	const int64_t m_batchsize = 64;

	bool m_bDoubleDQN = true;
	torch::optim::Adam* m_pAdam=nullptr;
};

