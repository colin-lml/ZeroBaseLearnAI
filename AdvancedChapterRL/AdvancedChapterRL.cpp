// AdvancedChapterRL.cpp: 定义应用程序的入口点。
//
#include "pch.h"
#include "AdvancedChapterRL.h"

using namespace std;

int main()
{
	DeepQNetwork  deepQN;
	//deepQN.PlayCartPole(300);
	DuelingDQN duelingDQN;
	duelingDQN.PlayCartPole(200);

	cin.get();
	return 0;
}
