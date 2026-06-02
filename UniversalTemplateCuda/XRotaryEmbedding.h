#pragma once

const double base = 10000;
const double maxSeqLen = 6;


class XRotaryEmbeddingImpl : public torch::nn::Module
{
public:
	XRotaryEmbeddingImpl();
	torch::Tensor forward(const torch::Tensor& x);
private:

	torch::Tensor PrecomputeFreqs(const torch::Tensor& x);

	torch::Tensor ApplyRotary(torch::Tensor x, const torch::Tensor& freqsCis);
};

TORCH_MODULE(XRotaryEmbedding);
