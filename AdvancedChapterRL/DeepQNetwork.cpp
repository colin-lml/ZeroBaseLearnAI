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

void DeepQNetwork::DoubleDQN(int maxCount)
{
	m_bDoubleDQN = true;
	BaseAdvanced::Play(maxCount);

}
double DeepQNetwork::TakeAction(VectorDouble& s0, bool bPredict)
{
	int a = 0;
	if (!bPredict && m_xRandom.RandDouble(0, 1.0) < m_dbEpsilon)
	{
		a = m_xRandom.RandInt(0, 1);
	}
	else
	{
		torch::NoGradGuard no_grad;
		
		auto s = VectorDoubleTensor(s0, m_device);
		
		auto q = m_Qnet->forward(s);
		a = q.squeeze().argmax().item<int>();
	}

	return a;
}

void DeepQNetwork::CreateOptimizer(DQNQnet& model)
{
	torch::optim::AdamOptions opt(m_dbLR);
	opt.betas({ 0.9, 0.98 });
	opt.eps(1e-9);
	opt.weight_decay(0);
	m_pAdam = new torch::optim::Adam(model->parameters(), opt);
	
}

void DeepQNetwork::SyncTargetNet()
{
	CopyModuleParameters(*m_Qnet, *m_TargetQnet);
	
#if 0
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
#endif

}

void DeepQNetwork::TrainGenerateItem1(const QwItem& item)
{
	AddCartPoleDataList(item);

}

void DeepQNetwork::GenerateTrainData(int maxCount)
{
	
	if (m_bDoubleDQN)
	{
		cout << "Currently DoubleDQN" << endl;
	}
	else
	{
		cout <<"Currently DQN" << endl;
	}

	GetCartPoleDataList().clear();

	auto input = m_objEnv->GetStateDim();
	auto output = m_objEnv->GetActionDim();

	m_Qnet = DQNQnet(input, output);
	m_TargetQnet = DQNQnet(input, output);
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

void DeepQNetwork::TrainGenerateItem2(const QwList& vList)
{

	if (400 < vList.size())
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

		for (int i = 0; i < max; i++)
		{
			Update();
		}
		
	}
}

void DeepQNetwork::Update()
{

	ReplayBuffer dataTrain; 
	static int count = 1;

	auto samples = dataTrain.sample(m_batchsize);

	auto [s0, a, r, s1, done] = QwListToTensor(samples, m_device);
	
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
	m_pAdam->zero_grad();
	dqnloss.backward();
	m_pAdam->step();
	auto loss  = dqnloss.item<double>();
	if (count % 10 == 0)
	{
		SyncTargetNet();
		//cout << "dqnloss: " << loss << endl;
	}
	count++;
			
	
}
