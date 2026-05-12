// GpuMakeProject.cpp: 定义应用程序的入口点。
//
#include <locale>
#include "GpuMakeProject.h"
#include "DecodersOnly.h"
#include "LoadDataset.h"
#include <filesystem>

void TrainData(DecodersOnly& model, translatDatasetOnly& dataTrain, int64_t maxtrain, int64_t batchsize);
void TestData3(DecodersOnly& model, translatDatasetOnly& dataTest);


#ifdef __TestData__

#define max_train  (1000 * 1)
#define batchsize2  50
#define nHeadLen 64   // 单头维度 = 64（优先）

#else
#define max_train  1000//*5
#define batchsize2  80
#define nHeadLen 64   // 单头维度 = 64（优先）

#endif // __TestData__


int main()
{
	torch::manual_seed(10);
	std::locale loc = std::locale();
	string name  = (loc.name()==""|| loc.name() == "C") ? "GBK" : loc.name();
	
	
	gDType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	string device = (gDType == torch::kCUDA) ? "kCUDA" : "kCPU";

	cout <<"本地编码: "<< name << endl;
	cout <<"device: " << device << endl;

	translatDatasetOnly dataTrain;
	
	DeOnlyOptions opt;

#ifdef __TestData__
	opt.head = 2;
	opt.dmodel = nHeadLen * opt.head;
	opt.ffn = opt.dmodel * 4;
	opt.layers = opt.head;
#else

	opt.head = 4;
	opt.dmodel = nHeadLen * opt.head;
	opt.ffn = opt.dmodel * 4;
	opt.layers = opt.head;

#endif
	opt.max_len = 1000;
	opt.vocab_size = gVocabCount;

	if (opt.ffn < gVocabCount * 2)
	{
		opt.ffn = gVocabCount * 2;
	}

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


