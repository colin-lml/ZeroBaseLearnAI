#pragma once

#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
#include <fstream>



#define  dim_model   128
#define  dim_feed    256
#define  max_vocab_len  500
#define  max_train       100

#define PadId 0

typedef std::unordered_map<std::string, int64_t> TableVocab;
typedef std::vector<std::pair<int64_t, int64_t>>  WordList;


/// <翻译>
///  Welcome to PyTorch Tutorials  ---> 欢迎来到派托奇教程
///  Welcome to Machine Learning -----> 欢迎来到机器学习
/// </翻译>
extern TableVocab src_vocab;
extern TableVocab tgt_vocab;


extern  int64_t src_vocab_size;
extern  int64_t tgt_vocab_size;


std::vector<std::string> Split(const std::string& s);


std::string GetWordById(TableVocab& vocabId, int64_t dataid);


std::vector<int64_t> GetWordId(TableVocab& vocabId, std::string data);


WordList GetLoadDataWordId(std::pair<std::string, std::string> data);




class translatDataset : public torch::data::Dataset<translatDataset>
{
public:

    translatDataset()
    {
        wordCount.push_back(GetLoadDataWordId({ "Welcome to PyTorch Tutorials Pad Pad Pad Pad Pad","欢 迎 来 到 派 托 奇 教 程" }));
        wordCount.push_back(GetLoadDataWordId({ "Welcome to Machine Learning  Pad Pad Pad Pad","欢 迎 来 到 机 器 学 习" }));

    }


    torch::optional<size_t> size() const
    {
        return wordCount.size();
    }

    torch::data::Example<torch::Tensor, torch::Tensor>  get(size_t index) override
    {
        auto item = wordCount[index];
        std::vector<int64_t> tmpinput;
        std::vector<int64_t> tmptarget1;


        for each(auto& i in item)
        {
            tmpinput.push_back(i.first);
            tmptarget1.push_back(i.second);

        }

        auto input = torch::tensor(tmpinput, torch::kLong);
        auto target = torch::tensor(tmptarget1, torch::kLong);


        return { input, target };
    }

public:
    std::vector<WordList> wordCount;

};

std::pair<torch::Tensor, torch::Tensor>  CreateDecoderInputOutput(torch::Tensor data);


class PositionalEncodingImpl :public torch::nn::Module
{
public:
    PositionalEncodingImpl(int64_t d_model, int64_t max_len)
    {
        _d_model = d_model;
        _max_len = max_len;
        _posEncode = torch::zeros({ _max_len, _d_model }, torch::kFloat32);

        Encoding();

        register_buffer("posEncode", _posEncode);
    }


    torch::Tensor forward(torch::Tensor x)
    {
        if ((x.dim() == 2))
        {
            x = x.unsqueeze_(-2);
        }

        auto dim = x.size(0);

        // std::cout <<"pos " << _posEncode.slice(0, 0, dim).sizes() << std::endl;
         //std::cout <<"x " << x.sizes() << std::endl;
        x = x + _posEncode.slice(0, 0, dim);
        return  x;
    }

private:
    void Encoding()
    {
        auto pos = torch::arange(0, _max_len, torch::kFloat32).reshape({ _max_len, 1 });
        auto den_indices = torch::arange(0, _d_model, 2, torch::kFloat32);
        auto den = torch::exp(-den_indices * std::log(10000.0f) / _d_model);
        _posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, _d_model, 2) }, torch::sin(pos * den));
        _posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, _d_model, 2) }, torch::cos(pos * den));
        _posEncode.unsqueeze_(-2);

    }

public:
    torch::Tensor _posEncode;
    int64_t _d_model = dim_model;
    int64_t _max_len = max_vocab_len;
};

TORCH_MODULE(PositionalEncoding);



class FeedForwardNetImpl : public torch::nn::Module
{
public:
	FeedForwardNetImpl(int64_t dim = 512, int64_t dff = 2048)
	{

		ffn = register_module("SeqFFN", torch::nn::Sequential(torch::nn::Linear(dim, dff),
			torch::nn::GELU(),
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

	//q k v : [seq, batch, dim]
	torch::Tensor forward(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, torch::Tensor mask = {})
	{
		auto q1 = Q->forward(q);
		auto k1 = Q->forward(k);
		auto v1 = Q->forward(v);

		return ScaledDotProductAttention(q1, k1, v1, mask);
	}


private:
	void InitQKV(int64_t dim, int64_t head)
	{

		auto linear = torch::nn::LinearOptions(dim, dim).bias(false);

		Q = register_module("q", torch::nn::Linear(linear));
		K = register_module("k", torch::nn::Linear(linear));
		V = register_module("v", torch::nn::Linear(linear));
		Wo = register_module("Wo", torch::nn::Linear(linear)); // 输出投影

		norm_fact = 1.0 / sqrt(dim);

		Dk = dim / head;
		H = head;
	}

	/// q: [seq, batch, dim]
	torch::Tensor ScaledDotProductAttention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor& mask)
	{
		/// k==v q可以不等于 k v
		auto seq = q.size(0);
		auto batch = q.size(1);
		auto dim = q.size(2);

		auto seq2 = k.size(0);
		auto batch2 = k.size(1);
		auto dim2 = k.size(2);

		q = q.view({ seq,batch,H,Dk }); //q: [seq, batch, dim] ->   [S, B, H, Dk] 
		k = k.view({ seq2,batch2,H,Dk });
		v = v.view({ seq2,batch2,H,Dk });

		q = q.permute({ 1,2,0,3 }); //[S, B, H, Dk] --->[B, H, S, Dk]
		k = k.permute({ 1,2,0,3 });
		v = v.permute({ 1,2,0,3 });

		auto kt = k.permute({ 0,1,3,2 }); //kt:  [B, H, S, Dk] --> [B, H, Dk, S]

		//cout << "q\n" << q.sizes() << endl;
		//cout << "kt\n" << kt.sizes() << endl;


		auto attn_score = torch::matmul(q, kt);

		attn_score = attn_score * norm_fact;

		if (mask.defined())
		{
			///std::cout << "attn_score\n" << attn_score.sizes() << std::endl;
			//std::cout << "mask\n" << mask.sizes() << std::endl;
			attn_score += mask;
		}

		attn_score = torch::softmax(attn_score, -1); /// attn_score: [B, H, S, S]

		auto out = torch::matmul(attn_score, v); // [B, H, S, S] * [B, H, S, Dk]  ->  out: [B, H, S, Dk]
		out = out.transpose(1, 2).contiguous().view({ seq,batch, dim }); //  [B, H, S, Dk] --> [B, S, H, Dk] -> [seq,batch, dim]
		//cout <<"out\n" << out.squeeze() << endl;
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


torch::Tensor generate_square_subsequent_mask(int64_t sz);