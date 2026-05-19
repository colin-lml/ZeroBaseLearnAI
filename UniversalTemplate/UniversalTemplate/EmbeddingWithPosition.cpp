#include "pch.h"
#include "EmbeddingWithPosition.h"

EmbeddingWithPositionImpl::EmbeddingWithPositionImpl(int64_t dim, int64_t numWord)
{
	m_embWord = register_module("embWord", torch::nn::Embedding(torch::nn::EmbeddingOptions(numWord, dim)));
}

 //// x [bath, seq]
torch::Tensor EmbeddingWithPositionImpl::forward(torch::Tensor x)
{
    auto B = x.size(0);
    auto S = x.size(1);
    auto D = m_embWord->options.embedding_dim();
    x = m_embWord->forward(x) * std::sqrt(D);
    auto p = PositionEncoding(S, D);
    x = x + p;
    return x;
}

/// 
torch::Tensor EmbeddingWithPositionImpl::PositionEncoding(int64_t seqLen, int64_t dim)
{
    auto posEncoding = torch::zeros({ seqLen, dim }, torch::kFloat32);

    auto pos = torch::arange(0, seqLen, torch::kFloat32).reshape({ seqLen, 1 });
    auto den_indices = torch::arange(0, dim, 2, torch::kFloat32);
    auto den = torch::exp(-den_indices * std::log(10000.0f) / dim);
    posEncoding.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, dim, 2) }, torch::sin(pos * den));
    posEncoding.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, dim, 2) }, torch::cos(pos * den));
    posEncoding.unsqueeze_(0);
    return pos;
}