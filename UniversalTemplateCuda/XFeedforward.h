#pragma once


class XFeedforwardImpl : public torch::nn::Module
{
public:
	XFeedforwardImpl(int64_t dim, int64_t feedforward, bool bias = false);

	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Linear m_w1{ nullptr };
	torch::nn::Linear m_w2{ nullptr };
};
TORCH_MODULE(XFeedforward);
