#pragma once

using VectorDouble2D = std::vector<std::vector<double>>;


class FreeModel
{
public:
	FreeModel();
	void NStepSarsaIteration(int nStep=3,int maxCount = 500);
	void SarsaIteration(int maxCount = 500);
	void QLearningIteration(int maxCount = 500);
private:
	int TakeAction(int s1);
	void UpdateSarsa(int s0, int a0, double r, int s1, int a1);
	void UpdateNStepSarsa(const int nStep,int s0, int a0, double r, int s1, int a1, bool done);
	void UpdateOffPolicy(int s0, int a0, double r, int s1, int a1);
	void PrintPi();
	CliffWalkingEnv m_objEnv;
	VectorDouble2D m_2dQtable;

	ActionList m_2dPI;

	const double m_dbAlpha = 0.1;
	const double m_dbGamma = 0.9;
	const double m_dbEpsilon = 0.1;

	std::random_device m_rd;
	std::mt19937 m_gen;
	std::uniform_real_distribution<double> m_rand01;
	std::uniform_int_distribution<int> m_rand04;
	vector<int> m_vNStepStates;
	vector<int> m_vNStepActions;
	vector<double> m_vNStepRewards;

};

