// GpuMakeProject.cpp: 定义应用程序的入口点。
//

#include "GpuMakeProject.h"
using namespace std;


torch::DeviceType gDType = torch::kCUDA;


class EmbeddingWithPositionImpl : public torch::nn::Module
{
public:
    EmbeddingWithPositionImpl(int64_t d_model, int64_t vocab_size ,int64_t max_len)
    {
        m_iDmodel = d_model;
        m_iMaxlen = max_len;
        m_posEncode = torch::zeros({ m_iMaxlen, m_iDmodel }, torch::kFloat32);

        Encoding();
        m_emb = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, m_iDmodel)));
        register_buffer("posEncode", m_posEncode);
    }


    torch::Tensor forward(torch::Tensor x)
    {

        x = m_emb->forward(x);
        if ((x.dim() == 2))
        {
            x = x.unsqueeze_(-2);
        }
        auto dim = x.size(0);

        // std::cout <<"pos " << _posEncode.slice(0, 0, dim).sizes() << std::endl;
         //std::cout <<"x " << x.sizes() << std::endl;
        x = x + m_posEncode.slice(0, 0, dim);
        return  x;
    }

private:
    void Encoding()
    {
        auto pos = torch::arange(0, m_iMaxlen, torch::kFloat32).reshape({ m_iMaxlen, 1 });
        auto den_indices = torch::arange(0, m_iDmodel, 2, torch::kFloat32);
        auto den = torch::exp(-den_indices * std::log(10000.0f) / m_iDmodel);
        m_posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, m_iDmodel, 2) }, torch::sin(pos * den));
        m_posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, m_iDmodel, 2) }, torch::cos(pos * den));
        m_posEncode.unsqueeze_(-2);

    }


    torch::nn::Embedding m_emb{ nullptr };
    torch::Tensor m_posEncode;
    int64_t m_iDmodel = 0;
    int64_t m_iMaxlen = 0;
};

TORCH_MODULE(EmbeddingWithPosition);

struct DeOnlyOptions
{
    int64_t dmodel;
    int64_t head;
    int64_t ffn;
    int64_t layers;
    int64_t vocab_size;
    int64_t max_len;
};


class DecodersOnlyImpl : public torch::nn::Module
{

public:

    DecodersOnlyImpl(const DeOnlyOptions&  inOpt)
    {
        fc = register_module("fc", torch::nn::Linear(inOpt.dmodel, inOpt.vocab_size));

        m_emb = register_module("m_emb", EmbeddingWithPosition(inOpt.dmodel, inOpt.vocab_size, inOpt.max_len));
        
        moduleLayers = register_module("EncoderLayers", torch::nn::ModuleList());

        m_option = inOpt;

        for (int i = 0; i < inOpt.layers; i++)
        {
            torch::nn::TransformerEncoderLayerOptions opt(inOpt.dmodel, inOpt.head);
            opt.dim_feedforward(inOpt.ffn);
            //opt.dropout(0);
      
            moduleLayers->push_back(torch::nn::TransformerEncoderLayer(opt));
        }

        //torch::nn::TransformerEncoderLayer()
    }

public:
    EmbeddingWithPosition m_emb{ nullptr };
    torch::nn::ModuleList moduleLayers{ nullptr };
    torch::nn::Linear fc{ nullptr };
    DeOnlyOptions m_option;
};
TORCH_MODULE(DecodersOnly);


int main()
{
	gDType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

	//auto input = torch::tensor({ 0.050,0.10 }, torch::kDouble).to(gDType);
	///cout << input << endl;

	cin.get();
	return 0;
}
