#include "pch.h"
#include "FreeModel.h"

FreeModel::FreeModel():m_gen(m_rd()), m_rand01(0.0,1), m_rand04(0, MaxAction-1)
{
	auto S = m_objEnv.GetTableSize();
	m_2dQtable.resize(S, std::vector<double>(MaxAction, 0.0));	
	m_2dPI.resize(S, { 0,0,0,0 });
}


int FreeModel::TakeAction(int s0)
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





void FreeModel::UpdateSarsa(int s0, int a0, double r, int s1, int a1)
{
	double td = r + m_dbGamma * m_2dQtable[s1][a1] - m_2dQtable[s0][a0];
	m_2dQtable[s0] [a0] += m_dbAlpha * td;
}

void FreeModel::UpdateNStepSarsa(const int nStep,int s0, int a0, double r, int s1, int a1, bool done)
{
	m_vNStepStates.push_back(s0);
	m_vNStepActions.push_back(a0);
	m_vNStepRewards.push_back(r);

	if (m_vNStepStates.size() == nStep)
	{
		auto G = m_2dQtable[s1][a1];
		int n = m_vNStepStates.size()-1;
		for (int i=n; 0 <= i; i--)
		{
			G = m_vNStepRewards[i] + m_dbGamma * G;
			if (done == 1.0)
			{
				auto s = m_vNStepStates[i];
				auto a = m_vNStepActions[i];
				m_2dQtable[s][a] += m_dbAlpha * (G - m_2dQtable[s][a]);
			}

		}

		auto s = m_vNStepStates[0];
		auto a = m_vNStepActions[0];
		m_vNStepStates.erase(m_vNStepStates.begin());
		m_vNStepActions.erase(m_vNStepActions.begin());
		m_vNStepRewards.erase(m_vNStepRewards.begin());

		m_2dQtable[s][a] += m_dbAlpha * (G - m_2dQtable[s][a]);
	}

	if (done == 1.0)
	{
		m_vNStepStates.clear();
		m_vNStepActions.clear();
		m_vNStepRewards.clear();
	}

}


void FreeModel::UpdateOffPolicy(int s0, int a0, double r, int s1, int a1)
{
	const auto& row = m_2dQtable[s1];
	auto maxIdx = std::max_element(row.begin(), row.end());
	auto max = *maxIdx;

	double td = r + m_dbGamma * max - m_2dQtable[s0][a0];
	m_2dQtable[s0][a0] += m_dbAlpha * td;
}

/// <summary>
/// 蒙特卡洛
/// </summary>
/// <param name="maxCount"></param>

typedef struct _MonteCarlo
{
	int s;
	int a;
	double r;
	bool done;

}MonteCarlo;

 
void FreeModel::MonteCarloMethods(int maxCount)
{
	cout << "\n蒙特卡洛方法" << endl;
	
	VectorDouble2D v2dCount;
	auto S = m_objEnv.GetTableSize();
	v2dCount.resize(S, std::vector<double>(MaxAction, 0.0));
	m_2dQtable.resize(S, std::vector<double>(MaxAction, 0.0));
	m_2dPI.resize(S, { 0,0,0,0 });
	for (size_t i = 0; i < maxCount; i++)
	{
		vector<MonteCarlo> trajectory;

		auto s = m_objEnv.Reset();
		auto a = TakeAction(s);
		bool b = true;
		while (b)
		{

			auto info = m_objEnv.Step(a);  // { 1, idx, reward, done };
			
			b = GetTuple(3, info) == 0;
			auto s1 = GetTuple(1, info);
			auto r = GetTuple(2, info);
			auto a1 = TakeAction(s1);
			if (b && s1 == s && a == a1)
			{
				continue;
			}

			MonteCarlo item;
			item.s = s;
			item.a = a;
			item.r = r;
			item.done = !b;
			//trajectory.push_back(item);
			trajectory.emplace_back(item);
			a = a1;
			s = s1;
		}

		double G = 0.0;
		set<tuple<int, int>> visited;

		for (int n = trajectory.size()-1; 0<= n; n--)
		{
			G = trajectory[n].r + m_dbGamma * G;
			int s = trajectory[n].s;
			int a = trajectory[n].a;
			

#if 0
			v2dCount[s][a] += 1;
			m_2dQtable[s][a] += (G - m_2dQtable[s][a]) / v2dCount[s][a];
			
#else
			auto key = make_tuple(s, a);
			if (visited.find(key) == visited.end())
			{
				visited.insert(key);
				v2dCount[s][a] += 1;
				m_2dQtable[s][a] += (G - m_2dQtable[s][a]) / v2dCount[s][a];
			}
#endif
			

		}

	}

	PrintPi();
}

void FreeModel::SarsaIteration(int maxCount)
{
	cout << "\nSarsa算法" << endl;
	auto S = m_objEnv.GetTableSize();
	m_2dQtable.resize(S, std::vector<double>(MaxAction, 0.0));
	m_2dPI.resize(S, { 0,0,0,0 });

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
			UpdateSarsa(s, a, r, s1, a1);
			a = a1;
			s = s1;
		}
	}

	PrintPi();

}

void FreeModel::NStepSarsaIteration(int nStep, int maxCount)
{
	cout << "\n多步 Sarsa 算法" << endl;

	auto S = m_objEnv.GetTableSize();
	m_2dQtable.resize(S, std::vector<double>(MaxAction, 0.0));
	m_2dPI.resize(S, { 0,0,0,0 });

	m_vNStepStates.clear();
	m_vNStepActions.clear();
	m_vNStepRewards.clear();


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
			auto r = GetTuple(2, info);
			auto a1 = TakeAction(s1);
			UpdateNStepSarsa(nStep,s,a,r,s1,a1, !b);
			a = a1;
			s = s1;
		}
	}

	PrintPi();
}


void FreeModel::QLearningIteration(int maxCount)
{
	cout << "\nSQ-learning 算法" << endl;
	auto S = m_objEnv.GetTableSize();
	m_2dQtable.resize(S, std::vector<double>(MaxAction, 0.0));
	m_2dPI.resize(S, { 0,0,0,0 });

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
			auto r = GetTuple(2, info);
			auto a1 = TakeAction(s1);
			UpdateOffPolicy(s, a, r, s1, a1);
			a = a1;
			s = s1;
		}
	}

	PrintPi();
}


void FreeModel::PrintPi()
{
	cout << endl;
	vector<string> action;
	action.push_back("↑");
	action.push_back("↓");
	action.push_back("←");
	action.push_back("→");

	auto R = m_objEnv.GetRow();
	auto C = m_objEnv.GetCol();
	vector<double> vecValue(R*C, 0);


	for (int r = 0; r < R; r++)
	{
		for (size_t c = 0; c < C; c++)
		{
			int idx = r * C + c;
			if (0 < idx && idx < C - 1)
			{
				cout << " ****  ";
			}
			else if (idx == C - 1)
			{
				cout << " EEEE  ";
			}
			else
			{
				const auto& row = m_2dQtable[idx];
				auto maxIdx = std::max_element(row.begin(), row.end());
				auto v = *maxIdx;
				vecValue[idx] = v;
				


				cout << setw(4) << fixed << setprecision(2) << v << "  ";
			}

		}

		cout << endl;
	}

	cout << endl;
	
	MovePos move;
	///pair<int, int>:  x,y 
	move.push_back({ 0, -1 });      // 上   ^ 
	move.push_back({ 0, 1 });     // 下    v
	move.push_back({ -1, 0 });     // 左  < 
	move.push_back({ 1 , 0 });     // 右  >

	for (int r = 0; r < R; r++)
	{
		for (size_t c = 0; c < C; c++)
		{
			int idx = r * C + c;

			for (int i = 0; i < MaxAction; i++)
			{
				aout  m_nx = min(C - 1, max(0, c + move[i].first));
				m_ny = min(R - 1, max(0, r + move[i].second));
			}
		}
	}

	


	for (int r = 0; r < R; r++)
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