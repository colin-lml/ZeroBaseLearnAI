#include "pch.h"
#include "XDecoderLayer.h"

XDecoderLayerImpl::XDecoderLayerImpl(int64_t dim, int64_t head, int64_t feedforward, bool bias)
{
    torch::nn::LayerNormOptions normOpt({ dim });
    m_attention = register_module("multiAttention", XMultiHeadAttention(dim, head, bias));
    m_feedforward = register_module("feedforward", XFeedforward(dim, feedforward, bias));
    m_norm1 = register_module("norm1", torch::nn::LayerNorm(normOpt));
    m_norm2 = register_module("norm2", torch::nn::LayerNorm(normOpt));
}


// x [seq, batch, dim]
torch::Tensor XDecoderLayerImpl::forward(torch::Tensor& x, const torch::Tensor& mask, const torch::Tensor& paddingMask)
{

    /*

def forward(self,x): # x(B,T,D)
# ²Ð²îÁ´½Ó1£º norm1 + MultiHeadAttention(¶àÍ·) + dropout(Ëæ»ú¹éÁã)
res1_in = x  # ÊäÈë
attn_output = self.mha(self.norm1(res1_in))  # ²Ð²î1 + ¶àÍ·
res1_out = res1_in + attn_output

# ²Ð²îÁ´½Ó2£º norm2 + FeedForwardNetwork(Ç°À¡Ëõ·Å) + dropout(Ëæ»ú¹éÁã)
res2_in = res1_out  # ÊäÈë
ffn_output = self.ffn(self.norm2(res2_in))  # ²Ð²î2 + Ç°À¡Ëõ·Å
res2_out = res2_in + ffn_output  # (B,T,D)

return res2_out

*/

    auto attnOutput = m_attention->forward(x, x, x, mask,  paddingMask);

    auto y = m_norm1->forward(attnOutput + x);

    auto y2 = m_feedforward->forward(y);
  
    return m_norm2->forward(y2 + y);
}