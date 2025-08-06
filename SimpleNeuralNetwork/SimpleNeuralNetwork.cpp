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


double WeightedSum(double x, double w)
{
    double z =   x * w ;
    return z;
}

double Loss(double y, double targetY)
{
    return 0.5 * (y - targetY) * (y - targetY);
}

double YFunction(double x1, double w1, double x2, double w2, double b)
{
    double y = x1 * w1 + x2 * w2 + b;
    return y;
}


int main()
{
    double i1 = 0.05;
    double i2 = 0.1;
    double w1 = 0.15;
    double w2 = 0.2;
    double w3 = 0.25;
    double w4 = 0.3;
    double w5 = 0.4;
    double w6 = 0.45;
    double w7 = 0.5;
    double w8 = 0.55;
    double b1 = 0.35;
    double b2 = 0.6;
  
    double o1 = 0.01;
    double o2 = 0.99;

    double y1 = 0;
    double y2 = 0;
    double outy1 = 0;
    double outy2 = 0;

    double h1 = 0;
    double h2 = 0;
    double outh1 = 0;
    double outh2 = 0;

    double o1Loss = 0;
    double o2Loss = 0;
    double totalLoss = 0;



    h1 = YFunction(i1,w1,i2,w2,b1);
    h2 = YFunction(i1, w3, i2, w4, b1);
    outh1 = sigmoid(h1);
    outh2 = sigmoid(h2);

    y1 = YFunction(outh1, w5, outh2, w6, b2);
    y2 = YFunction(outh1, w7, outh2, w8, b2);

    outy1 = sigmoid(y1);
    outy2 = sigmoid(y2);

    o1Loss = Loss(outy1, o1);
    o2Loss = Loss(outy2, o2);
    totalLoss = o1Loss + o2Loss;




}
