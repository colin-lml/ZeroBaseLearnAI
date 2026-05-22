#include "pch.h"
#include "XMultiHeadAttention.h"

XMultiHeadAttentionImpl::XMultiHeadAttentionImpl(int64_t dim, int64_t head, bool bias)
{
    assert(dim % head == 0);

    auto linear = torch::nn::LinearOptions(dim, dim).bias(bias);

    Q = register_module("q", torch::nn::Linear(linear));
    K = register_module("k", torch::nn::Linear(linear));
    V = register_module("v", torch::nn::Linear(linear));
    W = register_module("W", torch::nn::Linear(linear)); // 输出投影

    m_dbNormFact = 1.0 / sqrt(dim);
    m_i64Dim = dim;
    m_i64Head = head;
}


//q k v : [seq, batch, dim]
torch::Tensor XMultiHeadAttentionImpl::forward(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& mask, const torch::Tensor& paddingMask)
{
    auto q1 = Q->forward(q);
    auto k1 = K->forward(k);
    auto v1 = V->forward(v);

    auto out= ScaledDotProductAttention(q1, k1, v1, mask, paddingMask);

    return W->forward(out);
}



//q\k\v : [seq, batch, dim]
torch::Tensor XMultiHeadAttentionImpl::ScaledDotProductAttention(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, const torch::Tensor& mask, const torch::Tensor& paddingMask)
{
    /// k==v ; q可以不等于 k v
    auto batch = q.size(1);
    auto seq = q.size(0);
    auto dim = q.size(2);

    auto batch2 = k.size(1);
    auto seq2 = k.size(0);
    auto dim2 = k.size(2);

    auto H = m_i64Head;
    auto D = m_i64Dim / m_i64Head;

    q = q.view({ seq, batch,   H, D }); // [s, b,  h, d]
    k = k.view({ seq2, batch2, H, D });
    v = v.view({ seq2, batch2, H, D });

    q = q.permute({ 1,2,0,3 }); //[s, b, h, d] --->[B, H, S, D]
    k = k.permute({ 1,2,0,3 });
    v = v.permute({ 1,2,0,3 });

    auto kt = k.permute({ 0,1,3,2 }); //kt:  [B, H, S, D] --> [B, H, D, S]

    //cout << "q\n" << q.sizes() << endl;
    //cout << "kt\n" << kt.sizes() << endl;

    auto attn_score = torch::matmul(q, kt);

    attn_score = attn_score * m_dbNormFact;

    if (mask.defined())
    {
        attn_score += mask;
    }

    if (paddingMask.defined())
    {
        attn_score.masked_fill_(paddingMask, -1e9);
    }

    attn_score = torch::softmax(attn_score, -1); /// attn_score: [B, H, S, S]

    auto out = torch::matmul(attn_score, v); // [B, H, S, S] * [B, H, S, D]  ->  out: [B, H, S, D]
    out = out.transpose(1, 2).contiguous().view({ seq,batch, dim2}); //  [B, H, S, D] --> [B, S, H, D] -> [batch,seq, dim]
 
   
    return out;
}
