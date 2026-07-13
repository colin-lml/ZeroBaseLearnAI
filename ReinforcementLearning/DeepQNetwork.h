#pragma once

using  CartPoleStateList  = vector<CartPoleState>;

using  VectorDouble = std::vector<double>;

/// <summary>
///QwItem:  state, action, reward, next_state, done
/// </summary>
using  QwItem = std::tuple<VectorDouble,int, double, VectorDouble, bool>;
using  QwList = vector<QwItem>;
using  QwList2D = vector<QwList>;
using  QwItemTensor = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;


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

QwItemTensor QwListToTensor(const QwList& item);

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
		std::mt19937 rng(std::random_device{}());

		int count = size();
		count = min(count, batchsize);

		output.reserve(count);

		auto& datas = GetCartPoleDataList();
		std::sample(datas.begin(), datas.end(), std::back_inserter(output), count, rng);
	
		return output;
	}

	

private:

	
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
	const int64_t inputN = 4;
	const int64_t hiddenN = 128;
	const int64_t outN = 2;

};

TORCH_MODULE(Qnet);



class DeepQNetwork
{
public:
	void PlayCartPole(int maxCount = 200, bool bDoubleDQN = false);
	

private:
	void TestData(int maxCount);
	void TrainData(int maxCount);
	int TakeAction(VectorDouble s0, bool bPredict = false);
	void TrainQnet(torch::optim::Adam& adam);

	void SyncTargetNet();
	torch::optim::Adam CreateOptimizer(Qnet& model);

	//const double m_dbAlpha = 0.1;
	const double m_dbGamma = 0.9;
	const double m_dbEpsilon = 0.1;
	const double m_dbLR = 2e-3;

	XRandom m_xRandomData;

	Qnet m_Qnet;
	Qnet m_TargetQnet;
	CartPoleEnv m_CartPoleEnv;

	const int m_nMinimalsize = 500;
	const int64_t m_batchsize = 64;

	bool m_bDoubleDQN = false;
};

