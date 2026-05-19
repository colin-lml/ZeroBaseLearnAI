#pragma once

class EmbeddingWithPositionImpl: public torch::nn::Module
{
public:
	EmbeddingWithPositionImpl(int64_t dim, int64_t wordSize);

	torch::Tensor forward(torch::Tensor x);

private:
    torch::Tensor PositionEncoding(int64_t seqLen, int64_t dim);

private:
	torch::nn::Embedding m_embWord{ nullptr };
};

TORCH_MODULE(EmbeddingWithPosition);

