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

double Loss_derivative(double computey, double truey)
{
    return computey - truey;
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

    double nety1 = 0;
    double nety2 = 0;
    double outy1 = 0;
    double outy2 = 0;

    double neth1 = 0;
    double neth2 = 0;
    double outh1 = 0;
    double outh2 = 0;

    double o1Loss = 0;
    double o2Loss = 0;
    double totalLoss = 0;
    int64_t ik = 0;
    int64_t ikMax = 10000 * 30;
    DWORD dwTime = GetTickCount();
    double accuracy = 0.0000006;
    double rate = 0.5;

    while (ik< ikMax)
    {
        ik++;
        neth1 = YFunction(i1,w1,i2,w2,b1);
        neth2 = YFunction(i1, w3, i2, w4, b1);
        outh1 = sigmoid(neth1);
        outh2 = sigmoid(neth2);

        nety1 = YFunction(outh1, w5, outh2, w6, b2);
        nety2 = YFunction(outh1, w7, outh2, w8, b2);

        outy1 = sigmoid(nety1);
        outy2 = sigmoid(nety2);

        o1Loss = Loss(outy1, o1);
        o2Loss = Loss(outy2, o2);
        totalLoss = o1Loss + o2Loss;
        if (abs(outy1 - o1) <= accuracy && abs(outy2 - o2) <= accuracy)
        {
            break;
        }

        double qOut1 =  Loss_derivative(outy1,o1);
        double qOut2 = Loss_derivative(outy2, o2);

        ///totalLoss 对 w5的导数
        double lossw5 = qOut1 * sigmoid_derivative_from_sigmoid(outy1) * outh1;
        double oldw5 = w5;
        w5 = w5 - rate * lossw5;

        /// totalLoss 对 w6的导数

        double lossw6 = qOut1 * sigmoid_derivative_from_sigmoid(outy1) * outh2;
        double oldw6 = w6;
        w6 = w6 - rate * lossw6;

        /// totalLoss 对 w7的导数
    
        double lossw7 = qOut2 * sigmoid_derivative_from_sigmoid(outy2)* outh1;

        double oldw7 = w7;
        w7 = w7 - rate * lossw7;

        double lossw8 = qOut2 * sigmoid_derivative_from_sigmoid(outy2) * outh2;
        double oldw8 = w8;
        w8 = w8 - rate * lossw8;


        ///totalLoss 对  w1的导数
        double lossw1 = (qOut1 * sigmoid_derivative_from_sigmoid(outy1) * oldw5 + qOut2 * sigmoid_derivative_from_sigmoid(outy2) * oldw7) * sigmoid_derivative_from_sigmoid(outh1) * i1;
        // (outy1 - o1) * sigmoid_derivative_from_sigmoid(outy1) * oldw5 *sigmoid_derivative_from_sigmoid(outh1) * i1  + (outy2 - o2) * sigmoid_derivative_from_sigmoid(outy2) * oldw7 * sigmoid_derivative_from_sigmoid(outh1) * i1;
        w1 = w1 - rate * lossw1;

        double lossw2 = (qOut1 * sigmoid_derivative_from_sigmoid(outy1) * oldw5 + qOut2 * sigmoid_derivative_from_sigmoid(outy2) * oldw7) * sigmoid_derivative_from_sigmoid(outh1) * i2;
        w2 = w2 - rate * lossw2;

        double lossw3 = (qOut1 * sigmoid_derivative_from_sigmoid(outy1) * oldw6 + qOut2 * sigmoid_derivative_from_sigmoid(outy2) * oldw8) * sigmoid_derivative_from_sigmoid(outh2) * i1;
        w3 = w3 - rate * lossw3;

        double lossw4 = (qOut1 * sigmoid_derivative_from_sigmoid(outy1) * oldw6 + qOut2 * sigmoid_derivative_from_sigmoid(outy2) * oldw8) * sigmoid_derivative_from_sigmoid(outh2) * i2;
        w4 = w4 - rate * lossw4;
    }

     dwTime = GetTickCount() - dwTime;
     cout <<"time: "<< dwTime << " ms, count: "<< ik <<", outy1:  "<< fixed << setprecision(8) << outy1<<", outy2:  "<< fixed << setprecision(8)<<outy2 << endl;
     cin >> dwTime;
}
