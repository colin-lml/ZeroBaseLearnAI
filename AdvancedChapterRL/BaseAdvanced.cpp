#include "pch.h"
#include "BaseAdvanced.h"

BaseAdvanced::BaseAdvanced()
{
	 m_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}


void BaseAdvanced::PlayCartPole(int maxCount)
{
	torch::manual_seed(12);

	GenerateTrainData(maxCount);

	TestData(maxCount);
}
void BaseAdvanced::TestData(int maxCount)
{
	cout << "TestData ....." << endl;

	maxCount = max(maxCount / 15, 10);

	for (size_t i = 0; i < maxCount; i++)
	{
		auto s0 = m_CartPoleEnv.reset();
		auto done = false;
		int64_t rewardCount = 0;
		int64_t step = 0;
		while (!done && step < 500)
		{
			auto a = TakeAction(s0, true);
			//{ state, reward, terminated, truncated };
			auto [s1, r, d, _] = m_CartPoleEnv.step(a);
			done = d;
			s0 = s1;
			rewardCount += r;
			step++;
		}
		cout << "count: " << i + 1 << " , rewardCount: " << rewardCount << endl;

	}
}

void BaseAdvanced::GenerateTrainData(int maxCount)
{
	cout << "GenerateTrainData ....." << endl;

	for (int i = 0; i < maxCount; i++)
	{
		auto s = m_CartPoleEnv.reset();
		auto done = false;
		int64_t rewardCount = 0;
		QwList vList;
		while (!done && rewardCount < 470)
		{
			auto a = TakeAction(s);
			//{ state, reward, terminated, truncated };
			auto [s1, r, b, t] = m_CartPoleEnv.step(a);
			done = b;
			rewardCount += r;
			//{state, action, reward, next_state, done}
			vList.push_back({ s, a, r, s1, b });
			TrainGenerateItem1(vList[vList.size()-1]);
			s = s1;
		}

		TrainGenerateItem2(vList);

		if (i % 20 == 0 || maxCount==(i+1))
		{
			cout << "train i: " << i << " / " << maxCount << " , rewardCount: " << rewardCount << endl;
		}
	}

	cout  << endl;
}

