#pragma once


class PolicyIteration
{
public:
	PolicyIteration();

	void Iteration();
	
private:
	void Evaluation();
	void Improvement();
	void PrintPi();
	double WeightedSum(const StateInfo& item);


	CliffWalkingEnv m_objEnv;

	vector<double> m_vecV;
	ActionList m_2dPI;
	const double m_dbTheta = 0.001;
	const double m_dbGamma = 0.9;
};

