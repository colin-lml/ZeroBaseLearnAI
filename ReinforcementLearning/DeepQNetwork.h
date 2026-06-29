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


class ReplayBuffer
{
public:

	size_t size() const
	{
		return GetCartPoleDataList().size();
	}

	QwList2D sample(int batchsize, int max=10)
	{
		QwList2D batch;
		QwList output;
		std::mt19937 rng(std::random_device{}());

		int count = size();

		output.reserve(count);

		auto& datas = GetCartPoleDataList();
		std::sample(datas.begin(), datas.end(), std::back_inserter(output), count, rng);

		
		for (size_t i = 0; i < output.size(); i += batchsize)
		{
			auto end = std::min(i + batchsize, output.size());
			QwList listItem;
			for (int k = i; k < end; k++) 
			{
				listItem.push_back(std::move(output[k]));
			}

			batch.push_back(std::move(listItem));
			if ((max* batchsize) < i)
			{
				break;
			}
		}


		return batch;
	}

	torch::Tensor& VectorDoubleTensor(const VectorDouble& item)
	{
		auto S0 = torch::empty({ 1, 4 }, torch::kFloat32);
		auto* pS0 = S0.data_ptr<float>();
		std::copy(item.begin(), item.end(), pS0);

		return S0;
	}

	QwItemTensor QwListToTensor(const QwList& item)
	{
		int64_t n = item.size();

		auto S0 = torch::empty({ n, 4 }, torch::kFloat32);
		auto A = torch::empty({ n, 1 }, torch::kInt);
		auto R = torch::empty({ n, 1 }, torch::kFloat32);
		auto S1 = torch::empty({ n, 4 }, torch::kFloat32);
		auto Done = torch::empty({ n, 1 }, torch::kInt);
		auto* pS0 = S0.data_ptr<float>();
		auto* pS1 = S1.data_ptr<float>();

		for (int64_t i = 0; i < n; ++i)
		{
			auto [s,a,r,s1,d] = item[i];
			std::copy(s.begin(), s.end(), pS0 + i * 4);
			A[i][0] = a;
			R[i][0] = r;
			std::copy(s1.begin(), s1.end(), pS1 + i * 4);
			Done[i][0] = d;
			//cout << s[0]<<" " << s[1] << " " << s[2] <<" " << s[3] << endl;
		}

		return { S0,A,R,S1, Done };
		
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
	void TestData(int maxCount);
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


	const int m_nMinimalsize = 800;
	const int64_t m_batchsize = 100;
};

