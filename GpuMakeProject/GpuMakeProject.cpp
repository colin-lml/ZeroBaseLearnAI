// GpuMakeProject.cpp: 定义应用程序的入口点。
//
#include <locale>
#include "GpuMakeProject.h"
#include "DecodersOnly.h"
#include "LoadDataset.h"

void TrainData(DecodersOnly& model, translatDatasetOnly& dataTrain, int64_t maxtrain, int64_t batchsize);
void TestData3(DecodersOnly& model, translatDatasetOnly& dataTest);

#define max_train  1000*10
#define batchsize2  80



int main()
{
	std::locale loc = std::locale();
	string name  = (loc.name()==""|| loc.name() == "C") ? "GBK" : loc.name();
	
	
	gDType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	string device = (gDType == torch::kCUDA) ? "kCUDA" : "kCPU";

	cout <<"本地编码: "<< name << endl;
	cout <<"device: " << device << endl;

	translatDatasetOnly dataTrain;

	DeOnlyOptions opt;
	opt.dmodel = 256;
	opt.head = 8;
	opt.ffn = 1024;
	opt.layers = 1;
	opt.max_len = 1000;
	opt.vocab_size = gVocabCount;


	DecodersOnly model(opt);
	
	model->to(gDType);

	try
	{
		std::string model_path = "Decoder_Only_model3.pt";
		std::ifstream filem(model_path);
		bool bmodel = filem.is_open();
		if (!bmodel)
		{
			if (gDType == torch::kCUDA)
			{
				TrainData(model, dataTrain, max_train, batchsize2);
			}
			else
			{
				TrainData(model, dataTrain, max_train/3, batchsize2 * 4);
			}

			
			torch::save(model, model_path);
		}
		else
		{
			//model->to(torch::kCPU);
			torch::load(model, model_path);
			std::cout << "load model ...." << std::endl;
		}

		filem.close();

		TestData3(model, dataTrain);
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


