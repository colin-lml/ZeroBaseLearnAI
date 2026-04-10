#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>

#include "TransformerTestData.h"



class DecoderLayerImpl : public torch::nn::Module
{
public:
	DecoderLayerImpl(int64_t dim, int64_t head, int64_t dff)
	{
		torch::nn::LayerNormOptions normOpt({ dim });
		norm1 = register_module("norm1", torch::nn::LayerNorm(normOpt));
		norm2 = register_module("norm2", torch::nn::LayerNorm(normOpt));
		norm3 = register_module("norm3", torch::nn::LayerNorm(normOpt));
		ffn = register_module("ffn", FeedForwardNet(dim, dff));
		attention = register_module("attention", MultiHeadAttention(dim, head));
		attention2 = register_module("attention2", MultiHeadAttention(dim, head));
	}

	auto forward(torch::Tensor& tgt, torch::Tensor& memory, torch::Tensor tgtmask)
	{

		auto y = MaskAttention(tgt, tgtmask);

		//cout << "y\n" << y.sizes() << endl;
		//cout << "memory\n" << memory.sizes() << endl;

		auto y2 = attention2->forward(y, memory, memory);

		auto y3 = norm2->forward(y + y2); //  残差连接

		auto y4 = ffn->forward(y3);

		return norm3->forward(y3 + y4); //  残差连接
	}

private:

	torch::Tensor MaskAttention(torch::Tensor x, torch::Tensor mask)
	{
		auto y = attention->forward(x, x, x, mask);
		y = norm1->forward(x + y); //  残差连接
		return y;
	}

public:
	FeedForwardNet ffn{ nullptr };
	torch::nn::LayerNorm norm1{ nullptr }, norm2{ nullptr }, norm3{ nullptr };
	MultiHeadAttention attention{ nullptr };
	MultiHeadAttention attention2{ nullptr };
};

TORCH_MODULE(DecoderLayer);


class DecodersImpl : public torch::nn::Module
{
public:
	DecodersImpl(int64_t dim, int64_t head, int64_t ffn, int64_t layers)
	{
		moduleLayers = register_module("moduleLayers2", torch::nn::ModuleList());

		for (int i = 0; i < layers; i++)
		{
			moduleLayers->push_back(DecoderLayer(dim, head, ffn));
		}
	}

	auto forward(torch::Tensor& tgt, torch::Tensor& memory, torch::Tensor tgtmask)
	{

		for each(auto& item in * moduleLayers)
		{
			tgt = item->as<DecoderLayer>()->forward(tgt, memory, tgtmask);
		}

		return tgt;
	}

	torch::nn::ModuleList moduleLayers{ nullptr };

};

TORCH_MODULE(Decoders);


void DecoderOnlyMain()
{

}

