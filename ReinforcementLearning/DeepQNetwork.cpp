#include "pch.h"
#include "DeepQNetwork.h"

static QwList gCartPoleDataList;

QwList& GetCartPoleDataList()
{
	return gCartPoleDataList;
}

void AddCartPoleDataList(const QwItem& item)
{
	gCartPoleDataList.emplace_back(item);
}

void DeepQNetwork::PlayCartPole(int maxCount)
{
	TrainData(maxCount);
}

int DeepQNetwork::TakeAction(VectorDouble s0)
{
	int a = 0;
	if (m_xRandomData.RandDouble(0, 1.0) < m_dbEpsilon)
	{
		a = m_xRandomData.RandInt(0, 1);
	}
	else
	{
		a = m_xRandomData.RandInt(0, 1);
	}

	return a;
}



void DeepQNetwork::TrainData(int maxCount)
{
	for (int i = 0; i < maxCount; i++)
	{
		auto s = m_CartPoleEnv.reset();
		auto done = false;
		int64_t rewardCount = 0;
		while (!done)
		{
			auto a = TakeAction(s);
			//{ state, reward, terminated, truncated };
			auto [s1, r, b, t] = m_CartPoleEnv.step(a);
			done = b;
			rewardCount += r;
			//{state, action, reward, next_state, done}
			AddCartPoleDataList({ s,a,r,s1,done });

			if (m_nMinimalsize < GetCartPoleDataList().size())
			{
				TrainQnet();
			}

			s = s1;

		}

	}
}

void DeepQNetwork::TrainQnet()
{

}
