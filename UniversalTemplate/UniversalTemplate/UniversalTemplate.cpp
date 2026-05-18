// UniversalTemplate.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"


int main()
{
    std::vector<int64_t> tgtpad;

    tgtpad.push_back(0);
    tgtpad.push_back(5);
    tgtpad.push_back(6);
    tgtpad.push_back(7);

    torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong);
   // std::cout << "Hello World!\n";
    ///cout<<tgt<<endl;

    XBBPE xbbpe;


    std::cin.get();
}
