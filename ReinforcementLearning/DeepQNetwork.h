#pragma once

using  CartPoleStateList  = vector<CartPoleState>;

using  VectorDouble = std::vector<double>;

/// <summary>
///QwItem:  state, action, reward, next_state, done
/// </summary>
using  QwItem = std::tuple<VectorDouble,int, double, VectorDouble, bool>;
using  QwList = vector<QwItem>;

QwList& GetCartPoleDataList();
void AddCartPoleDataList(const QwItem& item);

class XRandom
{
public:
	XRandom(int64_t x = -1)
	{
		std::random_device rd;
		if (x < 0)
		{
			m_gen.seed(rd());
		}
		else
		{
			m_gen.seed(x);
		}

	}

	int RandInt(int min, int max)
	{
		std::uniform_int_distribution<int> rand(min, max);
		return rand(m_gen);
	}

	double RandDouble(double min, double max)
	{
		std::uniform_real_distribution<double> rand(min, max);
		return rand(m_gen);
	}
	

private:

	std::mt19937 m_gen;
};


class ReplayBuffer
{
public:

	size_t size() const
	{
		return GetCartPoleDataList().size();
	}

	QwList sample(size_t batchsize)
	{
		QwList output;
		std::mt19937 rng(std::random_device{}());

		int count = size();

		output.reserve(count);

		auto& datas = GetCartPoleDataList();
		std::sample(datas.begin(), datas.end(), std::back_inserter(output), count, rng);
		std::vector<QwList> batch;
		for (size_t i = 0; i < output.size(); i += batchsize)
		{
			auto end = std::min(i + batchsize, output.size());
			std::vector<QwList> item(output.begin() + i, output.begin() + end);
			batch.insert(batch.end(),batch.begin(), batch.end());
		}

	
		return output;
	}

	XRandom m_xRandomData;
};



class QnetImpl : public torch::nn::Module
{
public:
	QnetImpl()
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
	const int64_t inputN=4;
	const int64_t hiddenN=64;
	const int64_t outN=2;

};

TORCH_MODULE(Qnet);



class DeepQNetwork
{
public:
	void PlayCartPole(int maxCount = 500);
private:

	void TrainData(int maxCount);
	int TakeAction(VectorDouble s0);
	void TrainQnet(torch::optim::Adam& adam);

	void SyncTargetNet();
	torch::optim::Adam CreateOptimizer(Qnet& model);

	const double m_dbAlpha = 0.1;
	const double m_dbGamma = 0.9;
	const double m_dbEpsilon = 0.1;

	XRandom m_xRandomData;

	Qnet m_Qnet;
	Qnet m_TargetQnet;
	CartPoleEnv m_CartPoleEnv;


	const int m_nMinimalsize = 500;
	const int64_t m_batchsize = 100;
};

