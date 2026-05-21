#include "pch.h"
#include "XFeedforward.h"

XFeedforwardImpl::XFeedforwardImpl(int64_t dim, int64_t feedforward, bool bias)
{
	auto l1 = torch::nn::LinearOptions(dim, feedforward).bias(bias);
	auto l2 = torch::nn::LinearOptions(feedforward, dim).bias(bias);

	m_w1 = register_module("w1", torch::nn::Linear(l1));
	m_w2 = register_module("w2", torch::nn::Linear(l2));
}

torch::Tensor XFeedforwardImpl::forward(torch::Tensor x)
{
	x = m_w1->forward(x);

	x = torch::nn::GELU()->forward(x);

	return m_w2->forward(x);
}