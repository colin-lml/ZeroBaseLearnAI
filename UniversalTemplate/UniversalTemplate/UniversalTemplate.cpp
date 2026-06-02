// UniversalTemplate.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"


int main()
{
    torch::manual_seed(12);

    //XTrainPredict xTrainPredict;
   /// xTrainPredict.TestData();

    auto t2 = torch::ones({ 9, 4 }, torch::kFloat32);
    XRotaryEmbedding dd;
    cout <<"t2-rotary\n" << dd->forward(t2) << endl;

    std::cin.get();
}

