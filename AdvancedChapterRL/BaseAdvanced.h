#pragma once

class BaseAdvanced
{
public:
	BaseAdvanced();
	virtual ~BaseAdvanced() = default;

public:
	virtual void PlayCartPole(int maxCount);

protected:
	virtual void TestData(int maxCount);
	virtual void GenerateTrainData(int maxCount);
	virtual int TakeAction(VectorDouble& s0, bool bPredict = false) = 0;
	virtual void TrainGenerateItem1(const QwItem& item)=0;
	virtual void TrainGenerateItem2(const QwList& vList) = 0;

	CartPoleEnv m_CartPoleEnv;
	XRandom m_xRandom;

	double m_dbAlpha = 0.1;
	double m_dbGamma = 0.9;
	double m_dbEpsilon = 0.1;
	double m_dbLR = 2e-3;
	torch::DeviceType m_device;
};
