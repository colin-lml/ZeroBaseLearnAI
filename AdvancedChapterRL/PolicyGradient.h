#pragma once

class PolicyNetImpl : public torch::nn::Module
{
public:

	PolicyNetImpl() = default;

	PolicyNetImpl(int64_t input, int64_t output, int64_t hidden = 128)
	{
		m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
		m_fc2 = register_module("fc2", torch::nn::Linear(hidden, output));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(m_fc1->forward(x));
		x = m_fc2->forward(x);
		return torch::softmax(x, 1);
	}

	torch::nn::Linear m_fc1{ nullptr };
	torch::nn::Linear m_fc2{ nullptr };

};

TORCH_MODULE(PolicyNet);


class Categorical
{
public:
	
	explicit Categorical(const torch::Tensor& logits)
		: logits_(logits)
	{
		TORCH_CHECK(logits_.dim() == 2, "logits must be shape \[B, num\_actions\]");
	}

	torch::Tensor log_prob(const torch::Tensor& actions) const
	{
		auto acts = actions.to(torch::kLong);
		auto p_a = logits_.gather(1, acts);
		return torch::log(p_a + 1e-8);
	}

	torch::Tensor sample() const
	{
		return torch::multinomial(logits_, 1, true).squeeze(1);
	}

	// 计算 KL 散度（逐 batch 行）
	torch::Tensor kl_divergence(const Categorical& other) const
	{
		// 本分布的概率和 log 概率
		torch::Tensor p = torch::softmax(logits_, 1);
		torch::Tensor log_p = torch::log_softmax(logits_, 1);
		// 目标分布的 log 概率
		torch::Tensor log_q = torch::log_softmax(other.logits_, 1);
		// KL 散度 D_KL(P||Q) = sum_i p_i (log_p_i - log_q_i)
		torch::Tensor kl = (p * (log_p - log_q)).sum(1);
		return kl;
	}

private:
	torch::Tensor logits_;
};

#if 0
 class Categorical
{
public:
	explicit Categorical(const torch::Tensor& logits_or_probs, bool input_is_logits = true)
	{
		if (input_is_logits)
		{
			// 输入为 raw logits：内部计算 softmax / log_softmax（数值稳定）
			logits_ = logits_or_probs;
			log_probs_ = torch::log_softmax(logits_, /*dim=*/1);
			probs_ = torch::softmax(logits_, /*dim=*/1);
		}
		else
		{
			// 输入已是概率分布（每行和为1）
			probs_ = logits_or_probs;
			// 防止 log(0)
			log_probs_ = torch::log(probs_ + 1e-8);
		}

		TORCH_CHECK(log_probs_.dim() == 2, "Categorical expects a 2-D tensor [B, num_actions]");
	}

	// 返回与输入 actions 对应的对数概率，输出形状为 [B]
	torch::Tensor log_prob(const torch::Tensor& actions) const
	{
		auto acts = actions.to(torch::kLong);
		if (acts.dim() == 1)
			acts = acts.unsqueeze(1); // 变为 [B,1] 以便 gather 使用
		acts = acts.to(log_probs_.device());

		auto p_a = log_probs_.gather(1, acts); // [B,1]
		return p_a.squeeze(1);                 // 返回 [B]
	}

	// 采样动作，返回 shape [B] 的 LongTensor（按概率采样）
	torch::Tensor sample() const
	{
		// multinomial 期待概率分布（非 log-probs）
		auto samp = torch::multinomial(probs_, 1, /*replacement=*/true);
		return samp.squeeze(1);
	}

private:
	torch::Tensor logits_;    // 可选，raw logits（若构造时传入 logits）
	torch::Tensor probs_;     // 概率分布，shape [B, num_actions]
	torch::Tensor log_probs_; // log-probs，shape [B, num_actions]
};
#endif




class PolicyGradient : public BaseAdvanced
{
public:


protected:
	void GenerateTrainData(int maxCount) override;
	int TakeAction(VectorDouble& s0, bool bPredict = false) override;

	void TrainGenerateItem1(const QwItem& item) override {};
	void TrainGenerateItem2(const QwList& vList) override ;

	void CreateOptimizer(PolicyNet& model);

	PolicyNet m_Qnet;

	torch::optim::Adam* m_pAdam = nullptr;
};

