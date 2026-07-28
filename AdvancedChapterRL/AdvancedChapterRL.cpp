// AdvancedChapterRL.cpp: 定义应用程序的入口点。
//
#include "pch.h"
#include "AdvancedChapterRL.h"

using namespace std;

int main()
{
	DeepQNetwork  deepQN;
	//deepQN.PlayCartPole(200);
	//deepQN.DoubleDQN(200);
	DuelingDQN duelingDQN;
	//duelingDQN.PlayCartPole(200);

	PolicyGradient policyGradient;
	//policyGradient.PlayCartPole(500);

	ActorCritic actorCritic;
	actorCritic.PlayCartPole(1000);

	cin.get();
	return 0;
}
