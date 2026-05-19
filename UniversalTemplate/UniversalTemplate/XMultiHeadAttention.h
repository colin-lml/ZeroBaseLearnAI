#pragma once

class XMultiHeadAttentionImpl : public torch::nn::Module
{
public:
	XMultiHeadAttentionImpl(int64_t dim, int64_t head, bool bias = false);

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& mask, const torch::Tensor& paddingMask={});

private:
    torch::Tensor ScaledDotProductAttention(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, const torch::Tensor& mask, const torch::Tensor& paddingMask);

private:
    torch::nn::Linear Q{ nullptr };
    torch::nn::Linear K{ nullptr };
    torch::nn::Linear V{ nullptr };
    torch::nn::Linear W{ nullptr };

    int64_t m_i64Dim;
    int64_t m_i64Head;
    double m_dbNormFact = 0;
};

TORCH_MODULE(XMultiHeadAttention);
