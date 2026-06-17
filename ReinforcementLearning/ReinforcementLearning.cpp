// ReinforcementLearning.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"



int main()
{

    PolicyIteration policy;
   // policy.Iteration();
    double V[4] = { 0 };
    double v1 = 0, v2 = 0, v3 = 0;
    double B = 5e-6;
    for (int i = 0; i < 1000; i++)
    {
        v1 = 0.5*(-2 + V[1] + V[2]);
        v2 = 0.5 * (-2 + V[2] + V[3]);
        v3 = 0.5 * (-2 + V[1] + V[3]);


        if (abs(V[0] - v1) < B && abs(V[1] - v2) < B && abs(V[2] - v3) < B)
        {
            cout << "i: " << (i + 1) << "  v1: " << v1 << "  ";
            cout << ",v2:  " << v2 << "  ";
            cout << ",v3:  " << v3 << endl;
            break;
        }

        V[0] = v1;
        V[1] = v2;
        V[2] = v3;
        cout <<"i: " << (i + 1) << "  v1: " << v1 << "  ";
        cout << ",v2:  " << v2 <<"  " ;
        cout << ",v3:  " << v3 << endl;
    }


    cin.get();
}


