// ReinforcementLearning.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"



int main()
{
   

    PolicyIteration policy;
   // policy.Iteration();
   // policy.ValueIteration2();

    FreeModel td;
   // td.MonteCarloMethods();
     td.SarsaIteration();
    //td.NStepSarsaIteration();
    //td.QLearningIteration();
    
    cin.get();
}


