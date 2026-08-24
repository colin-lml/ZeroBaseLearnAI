#pragma once

#include <unordered_set>

QwList& GetReplayDataList();
void AddReplayDataList(const QwItem& item);


class ReplayBuffer
{
public:

	size_t size() const
	{
		return GetReplayDataList().size();
	}

	QwList sample(int batchsize)
	{
		QwList output;
		XRandom random;

		int count = size();
		count = min(count, batchsize);

		output.reserve(count);

		auto& datas = GetReplayDataList();
        if (count == 0)
		{
			return output;
		}
		/// std::sample(datas.begin(), datas.end(), std::back_inserter(output), count, random.GetGen());
		// Sampling random indices avoids traversing the entire replay buffer.
		std::unordered_set<size_t> indices;
		indices.reserve(static_cast<size_t>(count));
		const int lastIndex = static_cast<int>(size() - 1);
		while (indices.size() < static_cast<size_t>(count))
		{
            indices.insert(static_cast<size_t>(random.RandInt(0, lastIndex)));
		}
		for (auto index : indices)
		{
			output.push_back(datas[index]);
		}
	
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
	
	void DoubleDQN(int maxCount);
protected:

	void GenerateTrainData(int maxCount) override;
	double TakeAction(VectorDouble& s0, bool bPredict = false) override;
	void TrainGenerateItem1(const QwItem& item) override;
	void TrainGenerateItem2(const QwList& vList) override;

	void Update();

	void SyncTargetNet();
	void CreateOptimizer(DQNQnet& model);

	DQNQnet m_Qnet;
	DQNQnet m_TargetQnet;
	CartPoleEnv m_CartPoleEnv;

	const int m_nMinimalsize = 500;
	const int64_t m_batchsize = 80;

	bool m_bDoubleDQN = false;
	torch::optim::Adam* m_pAdam=nullptr;
};

