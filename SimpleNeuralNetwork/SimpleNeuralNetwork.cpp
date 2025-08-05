// SimpleNeuralNetwork.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"


double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x) 
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}


double sigmoid_derivative_from_sigmoid(double s) 
{
    return s * (1.0 - s);
}


double WeightedSum(double x, double w, double b)
{
    double z =   x * w + b;
    return sigmoid(z);
}
double Activate(double z)
{
    return sigmoid(z);
}

double Loss(double y, double targetY)
{
    return 0.5 * (y - targetY) * (y - targetY);
}

int main()
{
    double x = 1.5;
    double w = 0.8;
    double b = 0.2;
    double targetY = 0.8;
    double y = 0;
    y = WeightedSum(x, w, b);
    double z = Activate(y);
    double fLoss = Loss(y, targetY);


}
