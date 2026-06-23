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
	

}

void  CliffWalkingEnv::CreateTransitionMatrix()
{
	m_2dTransitionMatrix.resize(m_nRow * m_nCol, ActionList(MaxAction));
	MovePos move;
	///pair<int, int>:  x,y 
	move.push_back({ 0, -1 });      // 上   ^ 
	move.push_back({ 0, 1 });     // 下    v
	move.push_back({ -1, 0 });     // 左  < 
	move.push_back({ 1 , 0 });     // 右  >


	for (int i = 0; i < m_nRow; i++)
	{
		for (int j = 0; j < m_nCol; j++)
		{
			for (int k = 0; k < MaxAction; k++)
			{
				int idx = i * m_nCol + j;
				int x = min(m_nCol - 1, max(0, j + move[k].first));
				int y = min(m_nRow - 1, max(0, i + move[k].second));
				int nextidx = y * m_nCol + x;
				double done = 0;
				double reward = -1;

				if (i == 0 && 0 < j)
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

				m_2dTransitionMatrix[idx][k] = { 1,nextidx,reward,done };
			}

		}
	}
}

int GetActionNextPos(int idx, int action)
{
	int x = idx % COL;
	int y = idx / COL;
	auto p = GetActionNextPos(x,y, action);
	
	idx = p.second * COL + p.first;
	return idx;
}


pair<int, int> GetActionNextPos(int x,int y,int action)
{
	
	MovePos move;
	///pair<int, int>:  x,y 
	move.push_back({ 0, -1 });      // 上   ^ 
	move.push_back({ 0, 1 });     // 下    v
	move.push_back({ -1, 0 });     // 左  < 
	move.push_back({ 1 , 0 });     // 右  >
	
	x = min(COL - 1, max(0, x + move[action].first));
	y = min(ROW - 1, max(0, y + move[action].second));

	return {x,y};
}


StateInfo CliffWalkingEnv::Step(int action)
{

	auto p = GetActionNextPos(m_nx, m_ny, action);
	m_nx = p.first;
	m_ny = p.second;
	
	int idx = m_ny * m_nCol + m_nx;
	double done = 0;
	double reward = -1;
	if (0 < idx && idx < m_nCol)
	{
		reward = -100;
		done = 1;
		if (idx == (m_nCol - 1))
		{
			reward = 1;
		}
	}

	return { 1, idx, reward, done };
}

int CliffWalkingEnv::Reset()
{
	 m_nx = 0;
	 m_ny = 0;
	 return 0;
}