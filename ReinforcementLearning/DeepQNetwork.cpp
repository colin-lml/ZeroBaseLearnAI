#include "pch.h"
#include "DeepQNetwork.h"

static QwList gCartPoleDataList;

QwList& GetCartPoleDataList()
{
	return gCartPoleDataList;
}

void AddCartPoleDataList(const QwItem& item)
{
	gCartPoleDataList.emplace_back(item);
}



void DeepQNetwork::PlayCartPole(int maxCount)
{
	torch::manual_seed(12);


	TrainData(maxCount);
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
		ReplayBuffer d;
		auto s = d.VectorDoubleTensor(s0);
		auto q = m_Qnet->forward(s);
		a = m_xRandomData.RandInt(0, 1);
	}

	return a;
}

torch::optim::Adam DeepQNetwork::CreateOptimizer(Qnet& model)
{
	const double  LR = 2e-3;
	torch::optim::AdamOptions opt(LR);
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
		while (!done)
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

	}

	SyncTargetNet();



}
void DeepQNetwork::TestData(int maxCount)
{
	cout << "TestData ....." << endl;

	m_Qnet->eval();
	m_TargetQnet->eval();
	auto s = m_CartPoleEnv.reset();
	auto done = false;
	int64_t rewardCount = 0;
	while (!done)
	{
		auto a = TakeAction(s);
		//{ state, reward, terminated, truncated };
		auto [s1, r, b, t] = m_CartPoleEnv.step(a);
		done = b;
		rewardCount += r;
	}


}




void DeepQNetwork::TrainQnet(torch::optim::Adam& adam)
{
	ReplayBuffer dataTrain; 
	static int count = 1;

	auto samples = dataTrain.sample(m_batchsize);

	for (auto& item : samples)
	{
		auto [s0, a, r, s1, done] = dataTrain.QwListToTensor(item);
		//cout << s0 << endl;
		auto q = m_Qnet->forward(s0);
		q = q.gather(1, a);

		auto q1 = m_TargetQnet->forward(s1);

		auto [qv, _] = q1.max(1);
		q1 = qv.view({ -1,1 });
		auto qtargets = r + m_dbGamma * q1 * (1 - done);

		auto mseloss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
		auto dqnloss = mseloss->forward(q, qtargets);
		adam.zero_grad();
		dqnloss.backward();
		adam.step();

		if (count % 10 == 0)
		{
			SyncTargetNet();
			cout << "dqnloss: " << dqnloss.item<double>() << endl;
		}
		count++;

	}

}
