// UniversalTemplateCuda.cpp: 定义应用程序的入口点。
//

#include "UniversalTemplateCuda.h"


int main()
{
	
	torch::manual_seed(12);

	XTrainPredict xTrainPredict;

	xTrainPredict.TestData();

	return 0;
}
