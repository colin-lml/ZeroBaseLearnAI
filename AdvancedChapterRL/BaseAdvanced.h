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
	bool m_bEndGenerateTrain=false;

};


class Categorical
{
public:

	explicit Categorical(const torch::Tensor& values)
	{
		TORCH_CHECK(
			values.defined(),
			"Categorical: input tensor is undefined");

		TORCH_CHECK(
			values.dim() >= 1,
			"Categorical: input must have at least one dimension");

		TORCH_CHECK(
			values.size(-1) > 0,
			"Categorical: number of categories must be greater than zero");


		TORCH_CHECK(
			torch::all(values >= 0).item<bool>(),
			"Categorical: probabilities must be non-negative");

		auto probabilitySums = values.sum(-1, true);

		TORCH_CHECK(
			torch::all(probabilitySums > 0).item<bool>(),
			"Categorical: each probability distribution must have "
			"a positive sum");

		// 与 PyTorch Categorical 类似，对输入概率进行归一化。
		probs_ = values / probabilitySums;

		constexpr double epsilon = 1e-8;
		log_probs_ = torch::log(probs_.clamp_min(epsilon));

	}

	// 返回指定动作的 log π(a|s)。
	// actions 可以是 [B] 或 [B, 1]，返回形状为 [B]。
	torch::Tensor log_prob(const torch::Tensor& actions) const
	{
		auto indices = actions
			.to(log_probs_.device())
			.to(torch::kLong);

		if (indices.dim() == log_probs_.dim() &&
			indices.size(-1) == 1)
		{
			return log_probs_
				.gather(-1, indices)
				.squeeze(-1);
		}

		return log_probs_.gather(-1, indices.unsqueeze(-1)).squeeze(-1);
	}

	// 按类别概率采样，结果形状为 batch shape。
	torch::Tensor sample() const
	{
		const auto categoryCount = probs_.size(-1);
		auto flatProbs = probs_.reshape({ -1, categoryCount });

		auto samples = torch::multinomial(flatProbs, 1, true);

		std::vector<int64_t> outputShape;
		auto sizes = probs_.sizes();
		outputShape.insert(
			outputShape.end(),
			sizes.begin(),
			sizes.end() - 1);

		return samples.squeeze(-1).reshape(outputShape);
	}

	// 返回概率最大的类别，形状为 batch shape。
	torch::Tensor mode() const
	{
		return probs_.argmax(-1);
	}

	// H(π) = -Σ π(a) log π(a)。
	torch::Tensor entropy() const
	{
		return -(probs_ * log_probs_).sum(-1);
	}

	// 计算 D_KL(this || other)，返回每个 batch 对应的 KL 散度。
	torch::Tensor kl_divergence(const Categorical& other) const
	{
		TORCH_CHECK(
			probs_.sizes() == other.probs_.sizes(),
			"Categorical::kl_divergence: distribution shapes must match");

		TORCH_CHECK(
			probs_.device() == other.probs_.device(),
			"Categorical::kl_divergence: distributions must be on "
			"the same device");

		return (probs_ * (log_probs_ - other.log_probs_)).sum(-1);
	}

	const torch::Tensor& probs() const
	{
		return probs_;
	}

	// 与 torch.distributions.Categorical.logits 一致，返回归一化 logits，
	// 即 log-probabilities。
	const torch::Tensor& logits() const
	{
		return log_probs_;
	}

	int64_t num_events() const
	{
		return probs_.size(-1);
	}

private:
	torch::Tensor probs_;
	torch::Tensor log_probs_;
};

void CopyModuleParameters(const torch::nn::Module& src, torch::nn::Module& dst);
torch::Tensor ParametersToVector(const std::vector<torch::Tensor>& parameters);
torch::Tensor ParametersToVector(const torch::nn::Module& module);
void VectorToParameters(const torch::Tensor& flat, std::vector<torch::Tensor>& parameters);
void VectorToParameters(const torch::Tensor& flat, torch::nn::Module& module);