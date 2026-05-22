#pragma once

struct RMSNormImpl : torch::nn::Module 
{
    int dim;
    double eps;
    torch::Tensor weight{}, bias{};

    RMSNormImpl(int hidden_dim, double epsilon = 1e-6)
        : dim(hidden_dim), eps(epsilon) 
    {
        weight = register_parameter("weight", torch::ones({ dim }));
        bias = register_parameter("bias", torch::zeros({ dim }));
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        auto rms = x.pow(2).mean(-1, true).add(eps).sqrt();
        auto res = x / rms;
        return res * weight + bias;
    }
};
TORCH_MODULE(RMSNorm);




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
