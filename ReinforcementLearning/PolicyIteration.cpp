#include "pch.h"
#include "PolicyIteration.h"

PolicyIteration::PolicyIteration():m_objEnv(ROW, COL)
{
	ActionList item(1, { 0.25,0.25,0.25,0.25 });
	int m = m_objEnv.GetTableSize();

	m_vecV.resize(m, 0);
	m_2dPI.resize(m, { 0.25,0.25,0.25,0.25 });

}

void PolicyIteration::Evaluation()
{
	int S = m_objEnv.GetTableSize();
	const auto& P = m_objEnv.GetTransitionMatrix();

	while (true)
	{
		double maxDiff = 0;
		vector<double> newValue(S, 0);

		for (int s = 0; s < S; s++)
		{
			vector<double> qsaList;
			int a = 0;
			for (auto item : P[s])
			{
				auto p = get<0>(item);
				auto nextidx = get<1>(item);
				auto reward = get<2>(item);
				auto done = get<3>(item);
				double qsa = p * (reward + m_dbGamma * m_vecV[nextidx] * (1 - done));
				p = GetTuple(a, m_2dPI[s]);
				
				qsaList.push_back(qsa * p);
				a++;
			}
			


		}
	}
	


}