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
// torch::rms_norm();

#if 0 
class DeepseekV2RMSNorm(nn.Module) :
    def __init__(self, hidden_size, eps = 1e-6) :
    """
    DeepseekV2RMSNorm is equivalent to T5LayerNorm
    """
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

    def forward(self, hidden_states) :
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim = True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)
#endif


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
