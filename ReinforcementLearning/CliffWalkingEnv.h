#pragma once



using StateInfo = tuple<double, double, double, double>;
using ActionList = vector<StateInfo>;
using Vec2D = vector<ActionList>;
using MovePos = vector<pair<int, int>>;


double GetTuple(const size_t index, const StateInfo& info);
void  SetTuple(const size_t index, double d, StateInfo& info);

pair<int, int> GetActionNextPos(int x, int y, int action);
int GetActionNextPos(int idx, int action);

class CliffWalkingEnv
{
public:
	CliffWalkingEnv(int r= ROW,int c= COL);

	int GetRow()
	{
		return m_nRow;
	}
	int GetCol()
	{
		return m_nCol;
	}
	int GetTableSize()
	{
		return m_nRow * m_nCol;
	}

	Vec2D& GetTransitionMatrix()
	{
		return m_2dTransitionMatrix;
	}
	void  CreateTransitionMatrix();
	StateInfo Step(int action);
	int Reset();

private:
	int m_nRow;
	int m_nCol;
	Vec2D m_2dTransitionMatrix;
	int m_nx = 0;
	int m_ny = 0;
};

