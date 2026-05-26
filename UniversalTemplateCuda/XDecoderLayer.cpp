#include "pch.h"
#include "XDecoderLayer.h"

XDecoderLayerImpl::XDecoderLayerImpl(int64_t dim, int64_t head, int64_t feedforward, bool bias)
{
    torch::nn::LayerNormOptions normOpt({ dim });
    m_attention = register_module("multiAttention", XMultiHeadAttention(dim, head, false));
    m_feedforward = register_module("feedforward", XFeedforward(dim, feedforward, bias));
    m_norm1 = register_module("norm1", torch::nn::LayerNorm(normOpt));
    m_norm2 = register_module("norm2", torch::nn::LayerNorm(normOpt));
}


// x [seq, batch, dim]
torch::Tensor XDecoderLayerImpl::forward(torch::Tensor& x, const torch::Tensor& mask, const torch::Tensor& paddingMask)
{
    
    auto attnOutput = m_attention->forward(x, x, x, mask,  paddingMask);

    auto y = m_norm1->forward(attnOutput + x);

    auto y2 = m_feedforward->forward(y);
  
    return m_norm2->forward(y2 + y);
}