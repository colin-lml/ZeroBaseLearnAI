// AdvancedChapterRL.cpp: 定义应用程序的入口点。
//
#include "pch.h"
#include "AdvancedChapterRL.h"

using namespace std;

int main()
{
	DeepQNetwork  deepQN;
	deepQN.PlayCartPole(400);
	//deepQN.DoubleDQN(400);
	DuelingDQN duelingDQN;
	//duelingDQN.PlayCartPole(400);

	PolicyGradient policyGradient;
	//policyGradient.PlayCartPole(1000);

	ActorCritic actorCritic;
	//actorCritic.PlayCartPole(1000);

	cin.get();
	return 0;
}
