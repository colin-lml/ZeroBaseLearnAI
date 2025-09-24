// SimpleNeuralNetwork.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"


float sigmoid(float x)
{

    return 1.0 / (1.0 + std::exp(-x));
}
/* 
float sigmoid_derivative(float x) 
{
    float s = sigmoid(x);
    return s * (1.0 - s);
}
*/

float sigmoid_derivative(float s)
{
    return s * (1.0 - s);
}


float WeightedSum(float x, float w)
{
    float z =   x * w ;
    return z;
}

float Loss(float y, float targetY)
{
    
    return 0.5 * (y - targetY) * (y - targetY);
}

float Loss_derivative(float computey, float truey)
{
    return computey - truey;
}


float YFunction(float x1, float w1, float x2, float w2, float b)
{
    float y = x1 * w1 + x2 * w2 + b;
    return y;
}



int main()
{
    float i1 = 0.05;
    float i2 = 0.1;
    float w1 = 0.15;
    float w2 = 0.2;
    float w3 = 0.25;
    float w4 = 0.3;
    float w5 = 0.4;
    float w6 = 0.45;
    float w7 = 0.5;
    float w8 = 0.55;
    float b1 = 0.35;
    float b2 = 0.35;
    float b3 = 0.6;
    float b4 = 0.6;
  
    float o1 = 0.01;
    float o2 = 0.99;

    float nety1 = 0;
    float nety2 = 0;
    float outy1 = 0;
    float outy2 = 0;

    float neth1 = 0;
    float neth2 = 0;
    float outh1 = 0;
    float outh2 = 0;

    float o1Loss = 0;
    float o2Loss = 0;
    float totalLoss = 0;
    int64_t ik = 0;
    int64_t ikMax = 10000 * 30;
    DWORD dwTime = GetTickCount();
    float accuracy = 0.0000006;
    float rate = 0.5;

    while (ik< ikMax)
    {
        ik++;
        neth1 = YFunction(i1,w1,i2,w2,b1);
        neth2 = YFunction(i1, w3, i2, w4, b2);
        outh1 = sigmoid(neth1);
        outh2 = sigmoid(neth2);

        nety1 = YFunction(outh1, w5, outh2, w6, b3);
        nety2 = YFunction(outh1, w7, outh2, w8, b4);

        outy1 = sigmoid(nety1);
        outy2 = sigmoid(nety2);

        o1Loss = Loss(outy1, o1);
        o2Loss = Loss(outy2, o2);
        totalLoss = o1Loss + o2Loss;
        if (abs(outy1 - o1) <= accuracy && abs(outy2 - o2) <= accuracy)
        {
            break;
        }

        float qOut1 =  Loss_derivative(outy1,o1);
        float qOut2 = Loss_derivative(outy2, o2);

        ///totalLoss 对 w5的导数
        float lossw5 = qOut1 * sigmoid_derivative(outy1) * outh1;
        float oldw5 = w5;
        w5 = w5 - rate * lossw5;

        /// totalLoss 对 w6的导数

        float lossw6 = qOut1 * sigmoid_derivative(outy1) * outh2;
        float oldw6 = w6;
        w6 = w6 - rate * lossw6;

        /// totalLoss 对 w7的导数
    
        float lossw7 = qOut2 * sigmoid_derivative(outy2)* outh1;

        float oldw7 = w7;
        w7 = w7 - rate * lossw7;

        float lossw8 = qOut2 * sigmoid_derivative(outy2) * outh2;
        float oldw8 = w8;
        w8 = w8 - rate * lossw8;

        float losswb3 = qOut1 * sigmoid_derivative(outy1);
        b3 = b3 - rate * losswb3;

        float losswb4 = qOut2 * sigmoid_derivative(outy2);
        b4 = b4 - rate * losswb4;

        ///totalLoss 对  w1的导数
        float lossw1 = (qOut1 * sigmoid_derivative(outy1) * oldw5 + qOut2 * sigmoid_derivative(outy2) * oldw7) * sigmoid_derivative(outh1) * i1;
        // (outy1 - o1) * sigmoid_derivative_from_sigmoid(outy1) * oldw5 *sigmoid_derivative_from_sigmoid(outh1) * i1  + (outy2 - o2) * sigmoid_derivative_from_sigmoid(outy2) * oldw7 * sigmoid_derivative_from_sigmoid(outh1) * i1;
        w1 = w1 - rate * lossw1;

        float lossw2 = (qOut1 * sigmoid_derivative(outy1) * oldw5 + qOut2 * sigmoid_derivative(outy2) * oldw7) * sigmoid_derivative(outh1) * i2;
        w2 = w2 - rate * lossw2;

        float lossw3 = (qOut1 * sigmoid_derivative(outy1) * oldw6 + qOut2 * sigmoid_derivative(outy2) * oldw8) * sigmoid_derivative(outh2) * i1;
        w3 = w3 - rate * lossw3;

        float lossw4 = (qOut1 * sigmoid_derivative(outy1) * oldw6 + qOut2 * sigmoid_derivative(outy2) * oldw8) * sigmoid_derivative(outh2) * i2;
        w4 = w4 - rate * lossw4;

        float losswb1 = (qOut1 * sigmoid_derivative(outy1) * oldw5 + qOut2 * sigmoid_derivative(outy2) * oldw7)* sigmoid_derivative(outh1);
        b1 = b1 - rate * losswb1;
       
        float losswb2 = (qOut1 * sigmoid_derivative(outy1) * oldw5 + qOut2 * sigmoid_derivative(outy2) * oldw7) * sigmoid_derivative(outh2);
        b2 = b2 - rate * losswb1;


    }

     dwTime = GetTickCount() - dwTime;
     cout <<"time: "<< dwTime << " ms, count: "<< ik <<", outy1:  "<< fixed << setprecision(8) << outy1<<", outy2:  "<< fixed << setprecision(8)<<outy2 << endl;
     cin >> dwTime;

     /// 没有调整 b1,b2,b3,b4   time: 31 ms, count: 228767, outy1:  0.01000052, outy2:  0.98999940 
     /// 调整 b1,b2,b3,b4       time: 15 ms, count: 74038, outy1:  0.01000060, outy2:  0.98999952
     /// 调整 b1,b2,b3,b4       time: 0 ms,  count: 10000, outy1:  0.01158534, outy2:  0.98846063
}
