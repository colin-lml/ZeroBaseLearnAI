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


    torch::Tensor forward(torch::Tensor& x)
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
    int64_t dmodel = 512;
    int64_t head = 8;
    int64_t ffn = 2048;
    int64_t layers = 6;
    int64_t vocab_size = 300;
    int64_t max_len = 500;
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

    auto forward(torch::Tensor x)
    {
        // x [bath, seq]
        int64_t batch = x.size(0);
        int64_t seq = x.size(1);
        auto src_mask = generate_square_subsequent_mask(seq);

        auto tgt_key_padding_mask = (x == 0).to(torch::kBool);


        x = m_emb->forward(x);
        
        x = x.permute({ 1,0, 2 });
       // cout <<x.sizes()<< endl;

        for (auto& item :  *moduleLayers)
        {
            x = item->as<torch::nn::TransformerEncoderLayer>()->forward(x, src_mask, tgt_key_padding_mask);
            //std::cout << "tgt\n" << tgt.sizes() << std::endl;
            //tgt = item->as<torch::nn::TransformerEncoderLayer>()->forward(tgt);
        }

        return fc->forward(x);
    }

    torch::Tensor generate_square_subsequent_mask(int64_t sz)
    {
        auto mask = torch::triu(torch::ones({ sz, sz }, torch::kFloat32), 1);

        mask = mask.masked_fill(mask == 1, -std::numeric_limits<float>::infinity());
        return mask;
    }


public:
    EmbeddingWithPosition m_emb{ nullptr };
    torch::nn::ModuleList moduleLayers{ nullptr };
    torch::nn::Linear fc{ nullptr };
    DeOnlyOptions m_option;
};
TORCH_MODULE(DecodersOnly);

#define max_train  1000*5

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(4, 600);

	gDType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    double accuracy = 0.0003;

    DeOnlyOptions opt;
    opt.dmodel = 128;
    opt.head = 4;
    opt.ffn = 512;
    opt.layers = 1;
    opt.max_len = 800;
    opt.vocab_size = 610;

    float total_loss = 0;

    DecodersOnly model(opt);

    auto options = torch::nn::CrossEntropyLossOptions().ignore_index(0);
    torch::nn::CrossEntropyLoss loss_fn(options);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();
    // s: 1, e: 2 , p: 0

    vector<int64_t> in;
    vector<int64_t> lab;
    in.push_back(1);
    for (int i=3;i<320;i++)
    {
        int randomNumber = dis(gen);
        in.push_back(randomNumber);
        lab.push_back(randomNumber);
    }
   lab.push_back(2);

   for (int i = 0; i < 100; i++)
   {
       in.push_back(0);
       lab.push_back(0);
   }


    auto input = torch::tensor(in, torch::kLong).unsqueeze(0);
    auto lable = torch::tensor(lab, torch::kLong);
    
    cout << input.sizes()<< endl<<endl;
    cout << lable.sizes() << endl;

    for (int i = 0; i < max_train; i++)
    {
        auto output = model->forward(input);
        output = output.reshape({ -1, output.size(2) });

        auto loss = loss_fn(output, lable);
       
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        loss.backward();
        optimizer.step();

        total_loss = loss.item<float>();

        if (i % 10 == 0 || (i+1) == max_train)
        {
            cout << i + 1 << " , loss: " << total_loss << endl;
        }
        
        if (total_loss < accuracy)
        {
            cout << i + 1 << " , loss: " << total_loss << " , end... " << endl;
            break;
        }
    }

	//auto input = torch::tensor({ 0.050,0.10 }, torch::kDouble).to(gDType);
	///cout << input << endl;

	cin.get();
	return 0;
}
