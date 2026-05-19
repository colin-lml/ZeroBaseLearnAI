#pragma once

class XDecoderOnlyImpl : public torch::nn::Module
{
public:
	XDecoderOnlyImpl(int64_t numHeads, int64_t numWords);

	torch::Tensor forward(torch::Tensor x);
	void predict(VectorInt64& input, int64_t eos, int64_t maxSeq = 100);
private:
	torch::Tensor generate_square_subsequent_mask(int64_t sz);

	EmbeddingWithPosition m_embPos{ nullptr };
	torch::nn::ModuleList m_decoderLayers{ nullptr };
	torch::nn::Linear m_fc{ nullptr };
};

TORCH_MODULE(XDecoderOnly);
