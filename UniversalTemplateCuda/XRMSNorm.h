#pragma once

class XRMSNormImpl : public torch::nn::Module
{
public:
	XRMSNormImpl(int64_t dim, double eps = 1e-6);
	torch::Tensor forward(const torch::Tensor& x);

private:
	int64_t m_dim = 0;
	double m_eps = 0;
	torch::Tensor m_weight{nullptr};
};

TORCH_MODULE(XRMSNorm);