#include "pch.h"
#include "CliffWalkingEnv.h"


double GetTuple(const size_t  index, const StateInfo& info)
{
	double d = 0;
	switch (index)
	{
	case 0:
		d = get<0>(info);
		break;
	case 1:
		d = get<1>(info);
		break;
	case 2:
		d = get<2>(info);
		break;
	case 3:
		d = get<3>(info);
		break;
	default:
		break;
	}

	return d;
}

void  SetTuple(const size_t index, double d, StateInfo& info)
{
	switch (index)
	{
	case 0:
		get<0>(info) = d;
		break;
	case 1:
		get<1>(info) = d;
		break;
	case 2:
		get<2>(info) = d;
		break;
	case 3:
		get<3>(info) = d;
		break;
	default:
		break;
	}
}


CliffWalkingEnv::CliffWalkingEnv(int r, int c)
{
	m_nCol = c;
	m_nRow = r;

	m_2dTransitionMatrix.resize(r*c, ActionList(MaxAction));
	MovePos move;
	///pair<int, int>:  x,y 
	move.push_back({0, -1});      // ио   ^ 
	move.push_back({0, 1});     // об    v
	move.push_back({-1, 0});     // вС  < 
	move.push_back({1 , 0});     // ср  >


	for (int i = 0; i < m_nRow; i++)
	{
		for (int j = 0; j < m_nCol; j++)
		{
			for (int k = 0; k < MaxAction; k++)
			{
				int idx = i * m_nCol + j;
				int x = min(m_nCol-1, max(0, j + move[k].first));
				int y = min(m_nRow - 1, max(0, i + move[k].second));
				int nextidx = y * m_nCol + x;
				double done = 0;
				double reward = -1;

				if (i==0 && 0 < j)
				{
					reward = 0;
					done = 1;
					m_2dTransitionMatrix[idx][k] = { 1, idx, reward, done };
					continue;
				}

				if (y == 0 && 0 < x)
				{
					done = 1;
					if (x == m_nCol - 1)
					{
						reward = 1;
					}
					else
					{
						reward = -100;
					}

				}

				m_2dTransitionMatrix[idx][k] = {1,nextidx,reward,done};
			}
			
		}
	}

/* 
	for (int i = 0; i < m_nRow; i++)
	{
		for (int j = 0; j < m_nCol; j++)
		{
			int idx = i * m_nCol + j;
			cout <<"x: " <<i<<" y: " << j <<": "<<endl;
			for (auto& item : m_2dTransitionMatrix[idx])
			{
				cout << get<1>(item) << "," << get<2>(item) << "," << get<3>(item) ;
				cout << endl;
			}
			cout << endl;
		}

		cout << endl;
		cout << endl;
	}
*/
}
