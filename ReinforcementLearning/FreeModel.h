#pragma once

using VectorDouble2D = std::vector<std::vector<double>>;


class FreeModel
{
public:
	FreeModel();
	void MonteCarloMethods(int maxCount = 1500);
	void SarsaIteration(int maxCount = 500);
	void NStepSarsaIteration(int nStep=10,int maxCount = 1000);	
	void QLearningIteration(int maxCount = 500);
	void DynaQIteration(int maxCount = 500);
private:
	int TakeAction(int s1);
	void UpdateSarsa(int s0, int a0, double r, int s1, int a1);
	void UpdateNStepSarsa(const int nStep,int s0, int a0, double r, int s1, int a1, bool done);
	void UpdateOffPolicyQLearning(int s0, int a0, double r, int s1, int a1);
	void UpdateDynaQ(int s0, int a0, double r, int s1, int a1);
	void PrintPi();
	void UpdatePi(int idx, ActionList& m2dPI,const vector<double>& vecValue);
	CliffWalkingEnv m_objEnv;
	VectorDouble2D m_2dQtable;


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
	const int  m_nPlanning = 4;
	DynaQModelMap m_mapDynaQModel;

};

