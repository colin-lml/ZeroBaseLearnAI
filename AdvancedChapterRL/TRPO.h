#pragma once



class TRPO : public BaseAdvanced
{
public:
	

private:

	torch::Tensor ComputeAdvantage(double gamma, double lmbda, torch::Tensor td_delta);
	torch::Tensor ComputeSurrogateObj(torch::Tensor s, torch::Tensor a, torch::Tensor advantage, torch::Tensor logProbs, PolicyNet&  actorNet);
	void PolicyLearnUpdate(torch::Tensor s, torch::Tensor a, Categorical actionDists, torch::Tensor logProbs, torch::Tensor advantage);
	torch::Tensor ConjugateGradient(torch::Tensor objGrad, torch::Tensor s, Categorical actionDists);
	torch::Tensor HessianMatrixVectorProduct();

	void GenerateTrainData(int maxCount) override;

	int TakeAction(VectorDouble& s0, bool bPredict = false) override;

	void TrainGenerateItem1(const QwItem& item) override {};
	void TrainGenerateItem2(const QwList& vList) override;
	

	PolicyNet m_ActorNet;
	ValueNet m_CriticNet;

	const double m_dbCriticLR = 1e-2;
	const double m_dbklConstraint = 1e-4;
	const double m_dbLmbda = 0.95;
	torch::optim::Adam* m_pAdamCritic;
};

