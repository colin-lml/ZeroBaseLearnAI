// GpuMakeProject.cpp: 定义应用程序的入口点。
//
#include <locale>
#include "GpuMakeProject.h"
#include "DecodersOnly.h"
#include "LoadDataset.h"
#include <filesystem>

string GetCurrentPath();
bool TrainData(DecodersOnly& model, translatDatasetOnly& dataTrain, int64_t maxtrain, int64_t batchsize);
void TestData(DecodersOnly& model, translatDatasetOnly& dataTest);

void  LoadModel(DecodersOnly& model, const string& path);
void  SaveModel(DecodersOnly& model, const string& path);


#ifdef __TestData__

#define max_train  (1000 * 1)
#define batchsize2  10
#define nHeadLen 64   // 单头维度 = 64（优先）

#else
#define max_train  1000*30
#define batchsize2  60
#define nHeadLen 64   // 单头维度 = 64（优先）

#endif // __TestData__





int main()
{

	torch::manual_seed(10);
	std::locale loc = std::locale();
	string name  = (loc.name()==""|| loc.name() == "C") ? "GBK" : loc.name();
	
	gDType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	string device = (gDType == torch::kCUDA) ? "kCUDA" : "kCPU";

	cout <<"本地编码:     "<< name << endl;
	cout <<"train device: " << device << endl;

	translatDatasetOnly dataTrain;
	
	DeOnlyOptions opt;

#ifdef __TestData__
	opt.head = 2;
	opt.dmodel = nHeadLen * opt.head;
	opt.ffn = opt.dmodel * 4;
	opt.layers = opt.head;
#else

	opt.head = 2;
	opt.dmodel = nHeadLen * opt.head;
	opt.ffn = max(opt.dmodel * 4, gVocabCount * 2);
	opt.layers = opt.head;

#endif
	opt.max_len = 1000;
	opt.vocab_size = gVocabCount;

	DecodersOnly model(opt);

	
	
	try
	{
		std::string modelPath =  "Decoder_Only_model3.pt";

#ifdef __TestData__
		std::remove(model_path.c_str());
#endif // DEBUG

		std::ifstream filem(GetCurrentPath() + modelPath);
		bool bmodel = filem.is_open();

		if (!bmodel)
		{
			model->to(gDType);
			if (TrainData(model, dataTrain, max_train, batchsize2))
			{
				SaveModel(model, modelPath);
			}
			
		}
		else
		{
			gDType = torch::kCPU;
			model->to(gDType);
			LoadModel(model, modelPath);
			
			std::cout << "load model ...." << std::endl;
		}

		filem.close();

		TestData(model, dataTrain);
	}
	catch (const torch::Error& e)
	{
		std::cerr << " LibTorch 错误: " << e.what() << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << " 标准异常: " << e.what() << std::endl;
	}

	
	return 0;
}


