#include "pch.h"
#include "XRMSNorm.h"

XRMSNormImpl::XRMSNormImpl(int64_t dim, double eps)
{
	m_dim = dim;
	m_eps = eps;

	m_weight = register_parameter("weight", torch::ones({ dim }));
}

torch::Tensor XRMSNormImpl::forward(const torch::Tensor& x)
{
	return torch::rms_norm(x, { m_dim },m_weight, m_eps);
}