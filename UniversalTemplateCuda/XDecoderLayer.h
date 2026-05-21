#pragma once

class XDecoderLayerImpl : public torch::nn::Module
{
public:
	XDecoderLayerImpl(int64_t dim, int64_t head, int64_t feedforward, bool bias = false);
	torch::Tensor forward(torch::Tensor& x, const torch::Tensor& mask, const torch::Tensor& paddingMask);
private:

	torch::nn::LayerNorm m_norm1{ nullptr };
	torch::nn::LayerNorm m_norm2{ nullptr };
	XMultiHeadAttention m_attention{ nullptr };
	XFeedforward m_feedforward{ nullptr };
};

TORCH_MODULE(XDecoderLayer);
