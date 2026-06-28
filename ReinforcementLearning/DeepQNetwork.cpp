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
}

void DeepQNetwork::TrainQnet(torch::optim::Adam& adam)
{
	ReplayBuffer dataTrain; 
	auto samples = dataTrain.sample();

	for (auto& item : samples)
	{
		// state, action, reward, next_state, done
		auto [s0, a,r,s1,done] = item;
		auto x = torch::tensor( s0 , torch::kFloat).unsqueeze(0);
		cout << x.sizes() << endl;
		auto q = m_Qnet->forward(x);
		auto idx = torch::tensor({{a}}, torch::kLong);
		//cout <<"q1: " << q << endl;
		q = q.gather(1, idx);
		//cout<<"q2:" << q << endl;
		x = torch::tensor(s1, torch::kFloat).unsqueeze(0);
		auto q1 = m_TargetQnet->forward(x);
		//cout <<"q1: " << q1 << endl;
		auto[qv,_] = q1.max(1);
		q1 = qv.view({ -1,1 });
		//cout << "q2: " << q1 << endl;

	}


}
