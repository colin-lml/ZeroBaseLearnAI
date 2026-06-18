#include "pch.h"
#include "PolicyIteration.h"

PolicyIteration::PolicyIteration()
{
	m_objEnv.CreateTransitionMatrix();

	int m = m_objEnv.GetTableSize();

	m_vecV.resize(m, 0);
	m_2dPI.resize(m, { 0.25,0.25,0.25,0.25 });

}

void PolicyIteration::PolicyEvaluation()
{
	int S = m_objEnv.GetTableSize();
	const auto& P = m_objEnv.GetTransitionMatrix();
	int64_t count = 0;
	while (true)
	{
		double maxDiff = 0;
		vector<double> newValue(S, 0);
		
		for (int s = 0; s < S; s++)
		{
			int a = 0;
			for (auto& item : P[s])
			{
				auto qsa = WeightedSum(item);
				auto p = GetTuple(a, m_2dPI[s]);
				newValue[s] += qsa * p;
				a++;
			}

			maxDiff = max(maxDiff, abs(newValue[s] - m_vecV[s]));

		}

		m_vecV = newValue;

		if (maxDiff < m_dbTheta)
		{
			break;
		}
		count++;
	}

}
double PolicyIteration::WeightedSum(const StateInfo& item)
{
	auto p = GetTuple(0, item);
	auto nextidx = GetTuple(1, item);
	auto reward = GetTuple(2, item);
	auto done = GetTuple(3, item);
	auto qsa = p * (reward + m_dbGamma * m_vecV[nextidx] * (1 - done));
	
	return qsa;
}
void PolicyIteration::PolicyImprovement()
{
	int S = m_objEnv.GetTableSize();
	const auto& P = m_objEnv.GetTransitionMatrix();

	for (int s = 0; s < S; s++)
	{
		vector<double> qsaList;
		for (auto& item : P[s])
		{
			auto qsa = WeightedSum(item);
			qsaList.push_back(qsa);
		}

		double maxq = *std::max_element(qsaList.begin(), qsaList.end());
		int cntq = count(qsaList.begin(), qsaList.end(), maxq);
		double prob = 1.0 / cntq;
		for (int i = 0; i < qsaList.size(); i++)
		{
			double d = 0;
			if (qsaList[i] == maxq)
			{
				d = prob;
			}
			
			SetTuple(i, d, m_2dPI[s]);
			
		}
			
	}
}

void PolicyIteration::Iteration()
{
	while (true)
	{
		PolicyEvaluation();
		auto oldpi = m_2dPI;
		PolicyImprovement();
		if (oldpi == m_2dPI)
		{
			break;
		}
	}

	PrintPi();
}

void PolicyIteration::ValueIteration2()
{
	cout << "价值迭代算法" << endl;

	int S = m_objEnv.GetTableSize();
	m_vecV.resize(S, 0);
	m_2dPI.resize(S, { 0,0,0,0 });

	const auto& P = m_objEnv.GetTransitionMatrix();
	int64_t count = 0;
	while (true)
	{
		double maxDiff = 0;
		vector<double> newValue(S, 0);

		for (int s = 0; s < S; s++)
		{
			vector<double> listQsa;
			int a = 0;
			for (auto& item : P[s])
			{
				auto qsa = WeightedSum(item);
				listQsa.push_back(qsa);				
			}

			newValue[s] = *std::max_element(listQsa.begin(), listQsa.end());
			maxDiff = max(maxDiff, abs(newValue[s] - m_vecV[s]));

		}

		m_vecV = newValue;

		if (maxDiff < m_dbTheta)
		{
			break;
		}
		count++;
	}

	PolicyImprovement();

	PrintPi();

}




void PolicyIteration::PrintPi()
{
	cout << endl;
	vector<string> action;
	action.push_back("↑");
	action.push_back("↓");
	action.push_back("←");
	action.push_back("→");

	auto R = m_objEnv.GetRow();
	auto C= m_objEnv.GetCol();

	for (int r = 0; r < R; r++)
	{
		for (size_t c = 0; c < C; c++)
		{
			int idx = r * C + c;
			if (0 < idx && idx < C - 1)
			{
				cout << " ****  ";
			}
			else if (idx  == C - 1)
			{
				cout << " EEEE  ";
			}
			else
			{
				cout << setw(4) << fixed << setprecision(2) << m_vecV[idx] << "  ";
			}
			
		}

		cout << endl;
	}

	cout << endl;


	for (int r=0;r<R;r++)
	{
		for (size_t c = 0; c < C; c++)
		{
			int idx = r * C + c;
			if (0 < idx && idx < C - 1)
			{
				cout << " **** ";
			}
			else if (idx == C - 1)
			{
				cout << " EEEE ";
			}
			else
			{
				for (int i = 0; i < MaxAction; i++)
				{
					if (0 < GetTuple(i, m_2dPI[idx]))
					{
						cout << action[i];
					}
					else
					{
						cout << "o";
					}
				}
				cout << "  ";
			}

			
		}
		cout << endl;
	}
	
}