#include "pch.h"
#include "DeepQNetwork.h"

static QwList gCartPoleDataList;
static int gMaxCount = 10000;
QwList& GetCartPoleDataList()
{
	return gCartPoleDataList;
}

void AddCartPoleDataList(const QwItem& item)
{
	gCartPoleDataList.emplace_back(item);

	if (gMaxCount < gCartPoleDataList.size())
	{
		gCartPoleDataList.erase(gCartPoleDataList.begin());
	}
}


void DeepQNetwork::PlayCartPole(int maxCount, bool bDoubleDQN)
{
	torch::manual_seed(12);
	m_bDoubleDQN = bDoubleDQN;
	GetCartPoleDataList().clear();
	TrainData(maxCount);
}

torch::Tensor VectorDoubleTensor(const VectorDouble& item)
{
	auto S0 = torch::empty({ 1, 4 }, torch::kFloat32);
	auto* pS0 = S0.data_ptr<float>();
	std::copy(item.begin(), item.end(), pS0);

	return S0;
}


int DeepQNetwork::TakeAction(VectorDouble s0)
{
	int a = 0;
	if (m_xRandomData.RandDouble(0, 1.0) < m_dbEpsilon)
	{
		a = m_xRandomData.RandInt(0, 1);
	}
	else
	{
		
		auto s = VectorDoubleTensor(s0);
		
		auto q = m_Qnet->forward(s);
		a = q.squeeze().argmax().item<int>();
	}

	return a;
}

torch::optim::Adam DeepQNetwork::CreateOptimizer(Qnet& model)
{
	
	torch::optim::AdamOptions opt(m_dbLR);
	opt.betas({ 0.9, 0.98 });
	opt.eps(1e-9);
	opt.weight_decay(0);

	return torch::optim::Adam(model->parameters(), opt);
}

void DeepQNetwork::SyncTargetNet()
{
	string binPath = "tmpNetParameters.pt";
	{
		torch::serialize::OutputArchive archive;
		m_Qnet->save(archive);
		archive.save_to(binPath);
	}

	{
		torch::serialize::InputArchive archive;
		archive.load_from(binPath);
		m_TargetQnet->load(archive);
	}

}


void DeepQNetwork::TrainData(int maxCount)
{
	cout <<"TrainData....." << endl;
	auto adam = CreateOptimizer(m_Qnet);
	SyncTargetNet();

	for (int i = 0; i < maxCount; i++)
	{
		auto s = m_CartPoleEnv.reset();
		auto done = false;
		int64_t rewardCount = 0;

		while (!done && rewardCount < 470)
		{
			auto a = TakeAction(s);
			//{ state, reward, terminated, truncated };
			auto [s1, r, b, t] = m_CartPoleEnv.step(a);
			done = b;
			rewardCount += r;
			//{state, action, reward, next_state, done}
			AddCartPoleDataList({ s,a,r,s1,done });

			if (m_nMinimalsize < GetCartPoleDataList().size())
			{
				TrainQnet(adam);
			}

			s = s1;

		}

		if (i % 10 == 0)
		{
			cout << "train i: " << i <<" / "<< maxCount << " , rewardCount: " << rewardCount << endl;
		}
		
	}

	SyncTargetNet();
	TestData(maxCount);

}
void DeepQNetwork::TestData(int maxCount)
{
	cout << "TestData ....." << endl;

	m_Qnet->eval();
	m_TargetQnet->eval();

	auto s0 = m_CartPoleEnv.reset();
	auto done = false;
	int64_t rewardCount = 0;
	int64_t step = 0;
	while (!done && step < 500)
	{
		auto a = TakeAction(s0);
		//{ state, reward, terminated, truncated };
		auto [s1, r, d, _] = m_CartPoleEnv.step(a);
		done = d;
		s0 = s1;
		rewardCount += r;
		step++;
	}
	cout << "rewardCount: " << rewardCount << endl;

}




void DeepQNetwork::TrainQnet(torch::optim::Adam& adam)
{
	ReplayBuffer dataTrain; 
	static int count = 1;

	auto samples = dataTrain.sample(m_batchsize);

	auto [s0, a, r, s1, done] = dataTrain.QwListToTensor(samples);
	
	auto q = m_Qnet->forward(s0);
	q = q.gather(1, a);

	torch::Tensor q1 ;
	if (m_bDoubleDQN)
	{
		auto [_, idx] = m_Qnet->forward(s1).max(1); // max(): (Tensor values, Tensor indices)
		idx = idx.view({ -1,1 });
		q1 = m_TargetQnet->forward(s1).gather(1, idx);
		
	}
	else
	{
		q1 = m_TargetQnet->forward(s1);
		auto [qv, _] = q1.max(1);
		q1 = qv.view({ -1,1 });
	}

	auto qtargets = r + m_dbGamma * q1 * (1 - done);

	auto mseloss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
	auto dqnloss = mseloss->forward(q, qtargets);
	adam.zero_grad();
	dqnloss.backward();
	adam.step();
	auto loss  = dqnloss.item<double>();
	if (count % 10 == 0)
	{
		SyncTargetNet();
		//cout << "dqnloss: " << loss << endl;
	}
	count++;
			
	
}
