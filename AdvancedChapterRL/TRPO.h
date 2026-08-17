#pragma once



class TRPO : public BaseAdvanced
{
public:
	

private:
	void PolicyLearnUpdate(const torch::Tensor& s, const torch::Tensor& a, const Categorical& oldsDists,const torch::Tensor& oldLogProbs, const torch::Tensor& adv);
	torch::Tensor ComputeSurrogateObj(const torch::Tensor& s, const torch::Tensor& a, const torch::Tensor& adv, const torch::Tensor& oldLogProbs, PolicyNet& actorNet);
	torch::Tensor ComputeAdvantage(double gamma, double lmbda, torch::Tensor& td);
	torch::Tensor ConjugateGradient(const torch::Tensor& objGrad, const torch::Tensor& s, const Categorical& oldsDists);
	torch::Tensor HessianMatrixVectorProduct(const torch::Tensor& s, const Categorical& oldsDists, const torch::Tensor& v);
	torch::Tensor LineSearch(const torch::Tensor& s,const torch::Tensor& a, const torch::Tensor& adv, const torch::Tensor& oldLogProbs, const Categorical& oldsDists, const torch::Tensor& fullStep, bool& bUpdate);

	void GenerateTrainData(int maxCount) override;
	int TakeAction(VectorDouble& s0, bool bPredict = false) override;
	void TrainGenerateItem1(const QwItem& item) override {};
	void TrainGenerateItem2(const QwList& vList) override;
	
	PolicyNet m_ActorNet;
	ValueNet m_CriticNet;

	const double m_dbCriticLR = 1e-2;
	const double m_dbklConstraint = 5e-4;
	const double m_dbLmbda = 0.95;
	torch::optim::Adam* m_pAdamCritic;
};

