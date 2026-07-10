#include "pch.h"
#include "DuelingDQN.h"
QwList& GetCartPoleDataList();

torch::Tensor VectorDoubleTensor(const VectorDouble& item);


torch::optim::Adam DuelingDQN::CreateOptimizer(VANent& model)
{
	
	torch::optim::AdamOptions opt(m_dbLR);
	opt.betas({ 0.9, 0.98 });
	opt.eps(1e-9);
	opt.weight_decay(0);

	return torch::optim::Adam(model->parameters(), opt);
}


void DuelingDQN::SyncTargetNet()
{
	string binPath = "tmpDuelingDQNParameters.pt";
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

int DuelingDQN::TakeAction(VectorDouble s0)
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

void DuelingDQN::PlayCartPole(int maxCount)
{
	torch::manual_seed(22);
	GetCartPoleDataList().clear();

	TrainData(maxCount);

}

void DuelingDQN::TrainQnet(torch::optim::Adam& adam)
{
	ReplayBuffer dataTrain;
	static int count = 1;

	auto samples = dataTrain.sample(m_batchsize);

	auto [s0, a, r, s1, done] = dataTrain.QwListToTensor(samples);

	auto q = m_Qnet->forward(s0).gather(1, a);

	auto [q1, _]= m_TargetQnet->forward(s1).max(1);
	q1 = q1.view({ -1, 1 });
	
	auto qtargets = r + m_dbGamma * q1 * (1 - done);

	auto mseloss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
	auto dqnloss = mseloss->forward(q, qtargets);
	adam.zero_grad();
	dqnloss.backward();
	adam.step();

	if (count % 10 == 0)
	{
		SyncTargetNet();
	}
	count++;
}


void DuelingDQN::TrainData(int maxCount)
{
	cout << "DuelingDQN -> TrainData....." << endl;
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
			cout << "train i: " << i << " / " << maxCount << " , rewardCount: " << rewardCount << endl;
		}
	}

	TestData();
}

void DuelingDQN::TestData()
{
	cout << "TestData ....." << endl;

	m_Qnet->eval();


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
