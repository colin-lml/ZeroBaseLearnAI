// GpuMakeProject.cpp: 定义应用程序的入口点。
//

#include "GpuMakeProject.h"
#include "DecodersOnly.h"
#include "LoadDataset.h"

void TrainData(DecodersOnly& model, translatDatasetOnly& dataTrain, int64_t maxtrain, int64_t batchsize);


#define max_train  1000*2
#define batchsize2  50



int main()
{
   
	gDType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

	DeOnlyOptions opt;
	opt.dmodel = 200;
	opt.head = 10;
	opt.ffn = 1024;
	opt.layers = 1;
	opt.max_len = 100;
	opt.vocab_size = 105;


	DecodersOnly model(opt);
	translatDatasetOnly dataTrain;
	model->to(gDType);

	try
	{
		TrainData(model, dataTrain, max_train, batchsize2);
	}
	catch (const torch::Error& e)
	{
		std::cerr << " LibTorch 错误: " << e.what() << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << " 标准异常: " << e.what() << std::endl;
	}

	cin.get();
	return 0;
}


