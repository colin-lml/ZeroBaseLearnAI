#include "pch.h"
#include "TemporalDifference.h"

TemporalDifference::TemporalDifference():m_gen(m_rd()), m_rand01(0.0,1), m_rand04(0, MaxAction-1)
{
	auto S = m_objEnv.GetTableSize();
	m_2dQtable.resize(S, std::vector<double>(MaxAction, 0.0));	
}


int TemporalDifference::TakeAction(int s0)
{
	 
	double x = m_rand01(m_gen);
	int  action = 0;

	if (x < m_dbEpsilon)
	{
		action = m_rand04(m_gen);
	}
	else
	{
		const auto& row = m_2dQtable[s0];
		action = std::max_element(row.begin(), row.end()) - row.begin();
	}

	return action;

}

void TemporalDifference::Update(int s0, int a0, double r, int s1, int a1)
{
	double td = r + m_dbGamma * m_2dQtable[s1][a1] - m_2dQtable[s0][a0];
	m_2dQtable[s0] [a0] += m_dbAlpha * td;
}

void TemporalDifference::SarsaIteration(int maxCount)
{
	for (size_t i = 0; i < maxCount; i++)
	{
		auto s = m_objEnv.Reset();
		auto a = TakeAction(s);
		bool b = true;
		while (b)
		{
			auto info = m_objEnv.Step(a);  // { 1, idx, reward, done };
			b = GetTuple(3, info) == 0;
			auto s1 = GetTuple(1, info);
			auto r= GetTuple(2, info);
			auto a1 = TakeAction(s1);
			Update(s, a, r, s1, a1);
			a = a1;
			s = s1;
		}
	}
}