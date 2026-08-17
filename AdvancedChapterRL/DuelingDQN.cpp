#include "pch.h"
#include "DuelingDQN.h"
QwList& GetCartPoleDataList();



void DuelingDQN::CreateOptimizer(DuelingNet& model)
{
	
	torch::optim::AdamOptions opt(m_dbLR);
	opt.betas({ 0.9, 0.98 });
	opt.eps(1e-9);
	opt.weight_decay(0);
	m_pAdam  = new torch::optim::Adam(model->parameters(), opt);
	
}


void DuelingDQN::SyncTargetNet()
{

	CopyModuleParameters(*m_Qnet, *m_TargetQnet);

#if 0	
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
#endif

}

int DuelingDQN::TakeAction(VectorDouble& s0, bool bPredict)
{
	int a = 0;
	if (!bPredict && m_xRandomData.RandDouble(0, 1.0) < m_dbEpsilon)
	{
		a = m_xRandomData.RandInt(0, 1);
	}
	else
	{
		torch::NoGradGuard no_grad;
		auto s = VectorDoubleTensor(s0,m_device);
		auto q = m_Qnet->forward(s);
		a = q.squeeze().argmax().item<int>();
	}

	return a;
}

void DuelingDQN::TrainGenerateItem1(const QwItem& item)
{
	AddCartPoleDataList(item);

}

void DuelingDQN::TrainGenerateItem2(const QwList& vList)
{
	if (450 < vList.size())
	{
		m_bEndGenerateTrain = true;
		return;
	}

	if (m_nMinimalsize < GetCartPoleDataList().size())
	{
		int max = 10;
		if (100 < vList.size())
		{
			max = vList.size() / 2;
		}

		for (int i=0;i< max;i++)
		{
			Update();
		}
		
	}
}


void DuelingDQN::Update()
{
	ReplayBuffer dataTrain;
	static int count = 1;

	auto samples = dataTrain.sample(m_batchsize);

	auto [s0, a, r, s1, done] = QwListToTensor(samples, m_device);

	auto q = m_Qnet->forward(s0).gather(1, a);

	auto [q1, _]= m_TargetQnet->forward(s1).max(1);
	q1 = q1.view({ -1, 1 });
	
	auto qtargets = r + m_dbGamma * q1 * (1 - done);

	auto mseloss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
	auto dqnloss = mseloss->forward(q, qtargets);
	m_pAdam->zero_grad();
	dqnloss.backward();
	m_pAdam->step();

	if (count % 10 == 0)
	{
		SyncTargetNet();
	}
	count++;
}


void DuelingDQN::GenerateTrainData(int maxCount)
{
	cout << "Currently DuelingDQN" << endl;

	GetCartPoleDataList().clear();

	auto input = m_CartPoleEnv.GetStateDim();
	auto output = m_CartPoleEnv.GetActionDim();

	m_Qnet = DuelingNet(input, output);
	m_TargetQnet = DuelingNet(input, output);
	m_Qnet->to(m_device);
	m_TargetQnet->to(m_device);

	CreateOptimizer(m_Qnet);

	SyncTargetNet();

	m_Qnet->train();
	m_TargetQnet->train();

	BaseAdvanced::GenerateTrainData(maxCount);

	m_Qnet->eval();
	m_TargetQnet->eval();
	delete m_pAdam;
	m_pAdam = nullptr;
	
}

