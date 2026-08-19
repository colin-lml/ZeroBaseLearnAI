// AdvancedChapterRL.cpp: 定义应用程序的入口点。
//
#include "pch.h"
#include "AdvancedChapterRL.h"

using namespace std;

int main()
{
	DeepQNetwork  deepQN;
	//deepQN.Play(400);
	//deepQN.DoubleDQN(400);
	DuelingDQN duelingDQN;
	//duelingDQN.Play(400);

	PolicyGradient policyGradient;
	//policyGradient.Play(1000);

	ActorCritic actorCritic;
	//actorCritic.Play(1000);

	TRPO trpo;
	//trpo.Play(500);

	PPO ppo;
	//ppo.Play(500);
	DDPG ddpg;
	ddpg.Play(500);

	cin.get();
	return 0;
}
