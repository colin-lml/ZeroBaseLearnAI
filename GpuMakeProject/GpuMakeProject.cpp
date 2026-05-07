// GpuMakeProject.cpp: 定义应用程序的入口点。
//

#include "GpuMakeProject.h"
using namespace std;


torch::DeviceType gDType = torch::kCUDA;

static	int64_t  gBOS = 5001;
static	int64_t  gEOS = 5002;
static	int64_t  gPad = 5003;



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
        // x [bath, seq]
        x = m_emb->forward(x);

        // x [bath, seq, dim]

        auto dim = x.size(1);
       // cout << "x  " << x.sizes() << endl;
        //cout << "m_posEncode  "  << m_posEncode.sizes() << endl;
        //cout << "m_posEncode.slice  " << m_posEncode.slice(1, 0, dim).sizes() << endl;

        x = x + m_posEncode.slice(1, 0, dim);

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
        m_posEncode.unsqueeze_(0);
        // [bath,seq,dim]

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
        auto src_mask = generate_square_subsequent_mask(seq).to(gDType);

        auto tgt_key_padding_mask = (x == gPad).to(torch::kBool).to(gDType);

        x = m_emb->forward(x);
        
        x = x.permute({ 1,0, 2 });
       // cout <<x.sizes()<< endl;

        
       
        for (auto& item :  *moduleLayers)
        {
            x = item->as<torch::nn::TransformerEncoderLayer>()->forward(x, src_mask, tgt_key_padding_mask);

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

vector<pair<vector<int64_t>, vector<int64_t>>> MakeTestData(const int count=3)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 4900);
    vector<pair<vector<int64_t>, vector<int64_t>>> data;
    for (size_t i = 0; i < count; i++)
    {
        vector<int64_t> in;
        vector<int64_t> lab;
        in.push_back(gBOS);

        for (int i = 0; i < 3000; i++)
        {
            int randomNumber = dis(gen);
            in.push_back(randomNumber);
            lab.push_back(randomNumber);
        }
        lab.push_back(gEOS);

        for (int i = 0; i < 100; i++)
        {
            in.push_back(gPad);
            lab.push_back(gPad);
        }

        data.push_back({ in ,lab });

    }

    return data;
}

int main()
{

	gDType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    double accuracy = 0.0003;

    DeOnlyOptions opt;
    opt.dmodel = 256;
    opt.head = 8;
    opt.ffn = 1024;
    opt.layers = 1;
    opt.max_len = 10000;
    opt.vocab_size = 5005;

    
    DecodersOnly model(opt);

    model->to(gDType);

    auto options = torch::nn::CrossEntropyLossOptions().ignore_index(gPad);
    torch::nn::CrossEntropyLoss loss_fn(options);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();
   
    auto datas = MakeTestData(30);


    for (int i = 0; i < max_train; i++)
    {
        float total_loss = 0;

        for (auto& item :datas)
        {
            
            auto input = torch::tensor(item.first, torch::kLong).unsqueeze(0).to(gDType);
            auto lable = torch::tensor(item.second, torch::kLong).to(gDType);

            auto output = model->forward(input);
            output = output.reshape({ -1, output.size(2) });

            auto loss = loss_fn(output, lable);

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
        }


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


	cin.get();
	return 0;
}
