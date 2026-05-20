#pragma once


class XTrainPredict
{
public:
	XTrainPredict();
	void TestData();
private:
	bool TrainData(XDecoderOnly& model);
	bool LoadModel(XDecoderOnly& model);
	void SaveModel(XDecoderOnly& model, const string& path);
	void LoadTrainingBreakpoint(XDecoderOnly& model, torch::optim::Adam& optimizer, int& step);
	void SaveTrainingBreakpoint(XDecoderOnly& model, torch::optim::Adam& optimizer, int step);
	torch::optim::Adam CreateOptimizer(XDecoderOnly& model);

private:
	torch::DeviceType m_device;
	XBDataset m_xDataset;

	string m_strLogTrain = "train.log";
	string m_strCheckpoint = "checkpoint.pt";
	string m_strTmpModelPath = "";
 	const string m_strModelPath = "XUTModel.pt";

	const int64_t m_numHeads = 2;
	const double  LR = 2e-4;
	const int64_t m_maxtrain = 1000 * 10; //3w
	const int64_t m_batchsize = 50;
};

