#pragma once

class PPO : public BaseAdvanced
{
public:
	

private:

	torch::Tensor ComputeAdvantage(double gamma, double lmbda, torch::Tensor& td);


	void GenerateTrainData(int maxCount) override;
	double TakeAction(VectorDouble& s0, bool bPredict = false) override;

	void TrainGenerateItem1(const QwItem& item) override {};
	void TrainGenerateItem2(const QwList& vList) override;
	

	PolicyNet m_ActorNet;
	ValueNet m_CriticNet;

	
	const double m_dbActorLR = 1e-3;
	const double m_dbCriticLR = 1e-2;
	const double m_dbLmbda = 0.95;
	const double m_dbEps = 0.2;
	const int  m_nPPOEpochs = 10;

	torch::optim::Adam* m_pAdamActor;
	torch::optim::Adam* m_pAdamCritic;
};

