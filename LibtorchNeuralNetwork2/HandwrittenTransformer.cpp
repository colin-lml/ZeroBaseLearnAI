#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>
using namespace std;


#define  dim_model   128
#define  dim_feed    256
#define  max_vocab_len  500
#define  max_train       100

#define PadId 0

class FeedForwardNetImpl : public torch::nn::Module
{
public:
	FeedForwardNetImpl(int64_t dim=512, int64_t dff=2048)
	{

		ffn = register_module("SeqFFN", torch::nn::Sequential(torch::nn::Linear(dim, dff),
			torch::nn::ReLU(),
			torch::nn::Linear(dff, dim)
		));
	}

	auto forward(torch::Tensor x)
	{
		return ffn->forward(x);
	}

	torch::nn::Sequential ffn{};
	
};
TORCH_MODULE(FeedForwardNet);


class MultiHeadAttentionImpl : public torch::nn::Module
{
public:

	MultiHeadAttentionImpl(int64_t dim, int64_t head)
	{
		assert(dim % head == 0);

		InitQKV(dim, head);
	}

	//[seq, batch, dim]
	auto forward(torch::Tensor x, torch::Tensor mask = {})
	{
		assert(x.dim() == 3);

		//cout << "out\n" << x << endl;
		//cout << "Q\n" << Q << endl;

		auto q = Q->forward(x);
		auto k = K->forward(x);
		auto v = V->forward(x);

		auto out = ScaledDotProductAttention(q, k, v, mask);

		return out;
	}

private:
	void InitQKV(int64_t dim, int64_t head)
	{
		
		auto linear = torch::nn::LinearOptions(dim, dim).bias(false);

		Q = register_module("q", torch::nn::Linear(linear));
		K = register_module("k", torch::nn::Linear(linear));
		V = register_module("v", torch::nn::Linear(linear));
		Wo = register_module("Wo", torch::nn::Linear(linear)); // Êä³öÍ¶Ó°

		norm_fact = 1.0 / sqrt(dim);

		Dk = dim / head;
		H = head;
		
		//auto onesw = torch::eye(dim);
		//Q->weight.set_data(onesw);
		//K->weight.set_data(onesw);
		//V->weight.set_data(onesw);
		//Wo->weight.set_data(onesw);	
	}

	/// q: [seq, batch, dim]
	torch::Tensor ScaledDotProductAttention(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, torch::Tensor& mask)
	{
		auto seq = q.size(0);
		auto batch = q.size(1);
		auto dim = q.size(2);

		q = q.view({ seq,batch,H,Dk }); //q: [seq, batch, dim] ->   [S, B, H, Dk] 
		k = k.view({ seq,batch,H,Dk });
		v = v.view({ seq,batch,H,Dk });

		q = q.permute({ 1,2,0,3 }); //[S, B, H, Dk] --->[B, H, S, Dk]
		k = k.permute({ 1,2,0,3 });
		v = v.permute({ 1,2,0,3 });

		auto kt = k.permute({ 0,1,3,2 }); //kt:  [B, H, S, Dk] --> [B, H, Dk, S]

		auto attn_score = torch::matmul(q, kt);

		attn_score = attn_score * norm_fact;
	
		if (mask.defined())
		{
			attn_score += mask;
		}

		attn_score = torch::softmax(attn_score, -1); /// attn_score: [B, H, S, S]

		auto out = torch::matmul(attn_score, v); // [B, H, S, S] * [B, H, S, Dk]  ->  out: [B, H, S, Dk]
		out = out.transpose(1, 2).contiguous().view({ seq,batch, dim }); //  [B, H, S, Dk] --> [B, S, H, Dk] -> [seq,batch, dim]
		cout <<"out\n" << out.squeeze() << endl;
		out = Wo->forward(out);
		return out;
	}

	torch::nn::Linear Q{ nullptr };
	torch::nn::Linear K{ nullptr };
	torch::nn::Linear V{ nullptr };
	torch::nn::Linear Wo{ nullptr };

	double norm_fact = 0;
	int64_t Dk;
	int64_t H;
};
TORCH_MODULE(MultiHeadAttention);


class EncoderLayerImpl : public torch::nn::Module
{
public:
	EncoderLayerImpl(int64_t dim, int64_t head, int64_t dff)
	{
		torch::nn::LayerNormOptions normOpt({ dim });
		norm1 = register_module("norm1", torch::nn::LayerNorm(normOpt));
		norm2 = register_module("norm2", torch::nn::LayerNorm(normOpt));
		ffn = register_module("ffn", FeedForwardNet(dim, dff));
		attention = register_module("attention", MultiHeadAttention(dim, head));
	}

	auto forward(torch::Tensor x)
	{
		auto y = attention->forward(x);

		y = norm1->forward(x + y);

		auto y2 = ffn->forward(y);

		return norm2->forward(y + y2);
	}

	FeedForwardNet ffn{ nullptr };
	torch::nn::LayerNorm norm1{ nullptr }, norm2{ nullptr };
	MultiHeadAttention attention{ nullptr };
};

TORCH_MODULE(EncoderLayer);

class Encoders : public torch::nn::Module
{
public:
	Encoders(int64_t dim, int64_t head, int64_t ffn, int64_t layers)
	{
		moduleLayers = register_module("moduleLayers", torch::nn::ModuleList());

		for (int i = 0; i < layers; i++)
		{
			moduleLayers->push_back(EncoderLayer(dim, head, ffn));
		}
	}

	auto forward(torch::Tensor x)
	{
	
		for each(auto& item in *moduleLayers)
		{
			x = item->as<EncoderLayer>()->forward(x);
		}
		return x;
	}
	torch::nn::ModuleList moduleLayers{ nullptr };
	//torch::nn::ModuleList<EncoderLayer> layers{ nullptr };
};





void HandwrittenTransformerMain()
{
	torch::manual_seed(6);

	auto x = torch::tensor({
			{{1.0, 0.0, 0.0, 0.0}, // Welcome
			 {2.0, 0.0, 0.0, 0.0}, // to
			 {3.0, 0.0, 0.0, 0.0}, // Machine
			 {4.0, 0.0, 0.0, 0.0}, // Learning
			 {0.0, 0.0, 0.0, 0.0}, // Pad
			 {0.0, 0.0, 0.0, 0.0}  // Pad
			} }, torch::kFloat);

	Encoders ff(4,1, dim_feed,1);

	x = x.permute({ 1,0,2 });
	ff.forward(x);
}