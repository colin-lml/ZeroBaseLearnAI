#include "pch.h"
#include "BaseAdvanced.h"

BaseAdvanced::BaseAdvanced(bool bCartPole)
{
	m_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	if (bCartPole)
	{
		m_objEnv = new CartPoleEnv();
	}
	else
	{
		m_objEnv = new PendulumEnv();
	}
}

BaseAdvanced::~BaseAdvanced()
{
	delete m_objEnv;
}




void BaseAdvanced::PlayCartPole(int maxCount)
{
	maxCount = max(200, maxCount);

	torch::manual_seed(12);

	GenerateTrainData(maxCount);

	TestData(maxCount);
}
void BaseAdvanced::TestData(int maxCount)
{
	cout << "TestData ....." << endl;

	maxCount = max(maxCount / 15, 10);

	for (size_t i = 0; i < maxCount; i++)
	{
		auto s0 = m_objEnv->reset();
		auto done = false;
		double rewardCount = 0;
		int64_t step = 0;
		while (!done && step < 500)
		{
			auto a = TakeAction(s0, true);
			//{ state, reward, terminated, truncated };
			auto [s1, r, d, d2] = m_objEnv->step(a);
			done = d|| d2;
			s0 = s1;
			rewardCount+=r;
			step++;
		}
		cout << "count: " << i + 1 << " , rewardCount: " << rewardCount << endl;

	}
}

void BaseAdvanced::GenerateTrainData(int maxCount)
{
	cout << "GenerateTrainData ....." << endl;

	for (int i = 0; i < maxCount; i++)
	{
		auto s = m_objEnv->reset();
		auto done = false;
		double rewardCount = 0;
		int step = 0;	
		QwList vList;
		while (!done && step < m_maxMewardCount)
		{
			auto a = TakeAction(s);
			//{ state, reward, terminated, truncated };
			auto [s1, r, b, t] = m_objEnv->step(a);
			done = b||t;
			rewardCount+=r;
			//{state, action, reward, next_state, done}
			vList.push_back({ s, a, r, s1, b });
			TrainGenerateItem1(vList[vList.size()-1]);
			s = s1;
			step++;
		}

		TrainGenerateItem2(vList);

		if (70 < (i+1))
		{
			if ((i+1) % 20 == 0 || maxCount == (i + 1))
			{
				cout << "train i: " << i+1 << " / " << maxCount << " , rewardCount: " << rewardCount << endl;
			}
		}

		if (m_bEndGenerateTrain)
		{
			cout<< "train i: " << i + 1 << ", break Generate Train ####" << endl;
			break;
		}
	}

	cout  << endl;
}

// 将一个 module 的参数复制到另一个 module
void CopyModuleParameters(const torch::nn::Module& src, torch::nn::Module& dst)
{
	torch::NoGradGuard noGrad;

#if 0
	// 构建 src 参数的查找表： name -> tensor
	std::unordered_map<std::string, torch::Tensor> srcMap;
	for (const auto& item : src.named_parameters(/*recurse=*/true))
	{
		srcMap.emplace(item.key(), item.value());
	}

	// 遍历 dst 的参数，按 name 查找并 copy_
	for (auto& item : dst.named_parameters(/*recurse=*/true))
	{
		const auto& name = item.key();
		auto& dstTensor = item.value();
		auto it = srcMap.find(name);
		if (it == srcMap.end())
		{
			// 名称不匹配：跳过（或根据需要记录警告）
			continue;
		}
		auto srcTensor = it->second;
		// 保证类型/设备一致，然后原地复制
		if (srcTensor.device() != dstTensor.device())
		{
			srcTensor = srcTensor.to(dstTensor.device());
		}
		if (srcTensor.dtype() != dstTensor.dtype())
		{
			srcTensor = srcTensor.to(dstTensor.dtype());
		}
		dstTensor.copy_(srcTensor);
	}
#endif
	auto srcParams = src.parameters();
	auto dstParams = dst.parameters();
	TORCH_CHECK(srcParams.size() == dstParams.size(),"copy_module_parameters: parameter count mismatch!");
	for (size_t i = 0; i < srcParams.size(); ++i)
	{
		dstParams[i].copy_(srcParams[i]);
	}
}



torch::Tensor ParametersToVector(const std::vector<torch::Tensor>& parameters)
{
	std::vector<torch::Tensor> flatParameters;
	flatParameters.reserve(parameters.size());

	for (const auto& parameter : parameters)
	{
		if (!parameter.defined() || parameter.numel() == 0)
		{
			continue;
		}

		if (!flatParameters.empty())
		{
			const auto& firstParameter = flatParameters.front();

			TORCH_CHECK(
				parameter.device() == firstParameter.device(),
				"ParametersToVector: all parameters must be on the same device, "
				"but found ",
				firstParameter.device(),
				" and ",
				parameter.device());

			TORCH_CHECK(
				parameter.scalar_type() == firstParameter.scalar_type(),
				"ParametersToVector: all parameters must have the same dtype");
		}

		// 不调用 detach()，从而保留参数与返回向量之间的计算图。
		flatParameters.push_back(parameter.reshape({ -1 }));
	}

	if (flatParameters.empty())
	{
		return torch::empty({ 0 });
	}

	return torch::cat(flatParameters, 0);
}

torch::Tensor ParametersToVector(const torch::nn::Module& module)
{
	return ParametersToVector(module.parameters());
}

void VectorToParameters(const torch::Tensor& flat,std::vector<torch::Tensor>& parameters)
{
	TORCH_CHECK(flat.defined(), "VectorToParameters: flat tensor is undefined");

	// 参数赋值不应进入 autograd 计算图。
	torch::NoGradGuard noGrad;

	auto vector = flat.detach().reshape({ -1 });

	int64_t expectedNumel = 0;
	for (const auto& parameter : parameters)
	{
		if (parameter.defined())
		{
			expectedNumel += parameter.numel();
		}
	}

	TORCH_CHECK(
		vector.numel() == expectedNumel,
		"VectorToParameters: size mismatch, expected ",
		expectedNumel,
		" elements, but got ",
		vector.numel());

	int64_t offset = 0;

	for (auto& parameter : parameters)
	{
		if (!parameter.defined() || parameter.numel() == 0)
		{
			continue;
		}

		const auto parameterNumel = parameter.numel();

		auto parameterVector = vector
			.narrow(0, offset, parameterNumel)
			.to(parameter.options())
			.reshape(parameter.sizes());

		parameter.copy_(parameterVector);
		offset += parameterNumel;
	}

	TORCH_CHECK(
		offset == vector.numel(),
		"VectorToParameters: internal offset mismatch");
}

void VectorToParameters(const torch::Tensor& flat,torch::nn::Module& module)
{
	auto parameters = module.parameters();
	VectorToParameters(flat, parameters);
}
