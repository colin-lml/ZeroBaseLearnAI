#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>


#define  dim_model   128
#define  dim_feed    256
#define  max_vocab_len  500
#define  max_train       100

#define PadId 0

typedef const std::unordered_map< std::string, int64_t> TableVocab;
typedef std::vector<std::pair<int64_t, int64_t>>  WordList;


/// <翻译>
///  Welcome to PyTorch Tutorials  ---> 欢迎来到派托奇教程
///  Welcome to Machine Learning -----> 欢迎来到机器学习
/// </翻译>
TableVocab src_vocab =
{
    {"Pad",PadId},
    {"Welcome",1},
    {"to",2},
    {"PyTorch",3},
    {"Machine",4},
    {"Tutorials",5},
    {"Learning",6 }
};

TableVocab tgt_vocab =
{
    {"Pad",PadId},
    {"S",1},
    {"E",2},
    {"欢",3},
    {"迎",4},
    {"来",5},
    {"到",6},
    {"派",7},
    {"托",8},
    {"奇",9},
    {"教",10},
    {"程",11},
    {"机",12},
    {"器",13},
    {"学",14},
    {"习",15}
};


int64_t src_vocab_size = src_vocab.size();

int64_t tgt_vocab_size = tgt_vocab.size();


std::vector<std::string> Split(const std::string& s)
{
    std::vector<std::string> res;
    std::stringstream ss(s);
    std::string word;

    while (ss >> word)
    {
        res.push_back(word);
    }
    return res;
}

std::string GetWordById(TableVocab& vocabId,int64_t dataid)
{
    std::string Word = { "0" };
    for (auto& w : vocabId)
    {
        if (w.second == dataid)
        {
            Word = w.first;
            break;
        }
    }
    return Word;
}

std::vector<int64_t> GetWordId(TableVocab& vocabId,std::string data)
{
    std::vector<int64_t> input;
    for (auto ch : Split(data))
    {
        input.push_back(vocabId.at(ch));
    }
    return input;
}

WordList GetLoadDataWordId(std::pair<std::string, std::string> data)
{
    std::vector<int64_t> input = GetWordId(src_vocab,data.first);
    std::vector<int64_t> target = GetWordId(tgt_vocab,data.second);

    WordList item;
    for (int i = 0; i < input.size() && i < target.size(); i++)
    {
        item.push_back({ input.at(i),target.at(i) });
    }

    return item;
}



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


        return { input, target};
    }

public:
    std::vector<WordList> wordCount;

};

std::pair<torch::Tensor, torch::Tensor>  CreateDecoderInputOutput(torch::Tensor data)
{
   auto E = torch::tensor(GetWordId(tgt_vocab, "E"), torch::kLong).view({1,1});
   auto S = torch::tensor(GetWordId(tgt_vocab, "S"), torch::kLong).view({ 1,1 });

   auto  input = torch::cat({ S, data }, 1);
   auto  output = torch::cat({data,E}, 1);

   return { input ,output };
}

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


class TranslatorImpl : public torch::nn::Module
{
public:
    TranslatorImpl()
    {
        src_emb = register_module("src_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(src_vocab_size, dim_model)));
        tgt_emb_ = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(tgt_vocab_size, dim_model)));
        pos_encoder = register_module("pos_encoder", PositionalEncoding(dim_model, max_vocab_len));
        torch::nn::TransformerOptions opts;
        opts.nhead(2);
        opts.dim_feedforward(dim_feed);
        opts.num_decoder_layers(1);
        opts.num_encoder_layers(1);
        opts.dropout(0.0);
        opts.d_model(dim_model);
        transformer = register_module("transformer", torch::nn::Transformer(opts));
        fc = register_module("fc", torch::nn::Linear(dim_model, tgt_vocab_size));

    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt)
    {

        //[batch, seq]  --> [seq, batch]
        src = src.permute({ 1,0 });
        tgt = tgt.permute({ 1,0 });

        //std::cout << "input " << src << std::endl;
        src = src_emb->forward(src) * std::sqrt(dim_model);
        src = pos_encoder->forward(src);

        tgt = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
        tgt = pos_encoder->forward(tgt);

        // tgt & src: (seq, batch, dim)
        auto outs = transformer->forward(src, tgt);

        outs = fc->forward(outs);

        return outs;

    }

    torch::Tensor predict(torch::Tensor src)
    {
        auto srcemb = src_emb->forward(src) * std::sqrt(dim_model);
        srcemb = pos_encoder->forward(srcemb);
        auto memory = transformer->encoder.forward(srcemb);

        std::vector<int64_t> tgtpad = GetWordId(tgt_vocab, "S");
     
        int i = 0;
        while (true)
        {
            torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong);
            auto tgt_emb = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
            tgt_emb = pos_encoder->forward(tgt_emb);
            auto out = transformer->decoder.forward(tgt_emb, memory);
            out = fc->forward(out).squeeze(-2);
            auto next_token = out.argmax(-1);
            int64_t key = next_token[i].item<int64_t>();
            tgtpad.push_back(key);
            //tgtpad.insert(tgtpad.begin(), );
            if ("E" == GetWordById(tgt_vocab, key))
            {
                break;
            }
            i++;
        }
       
        //tgtpad.pop_back();
        return torch::tensor(tgtpad, torch::kLong);
    }


    torch::nn::Embedding src_emb{ nullptr };
    torch::nn::Embedding tgt_emb_{ nullptr };
    PositionalEncoding pos_encoder{ nullptr };
    torch::nn::Transformer transformer{ nullptr };
    torch::nn::Linear fc{ nullptr };
};
TORCH_MODULE(Translator);

void TestData(Translator& model);
void TrainData(Translator& model);





void TransformerMain()
{
    torch::manual_seed(4);

    std::string model_path = "translator_model.pt";
    Translator model;

    std::ifstream filem(model_path);
    bool bmodel = filem.is_open();
    if (!bmodel)
    {
        TrainData(model);
        torch::save(model, model_path);
    }
    else
    {
        torch::load(model, model_path);
        std::cout << "load model ...." << std::endl;
    }
    filem.close();


    TestData(model);

}




void TrainData(Translator& model)
{
    double accuracy = 0.03;

    auto datasetTrain = translatDataset().map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(1));
    torch::nn::CrossEntropyLoss loss_fn;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();
    std::cout << "训练模型" << std::endl;

    for (int i = 0; i < max_train; i++)
    {
        float total_loss = 0;
        for (auto& item : *train_data_loader)
        {
            /// item: [batch, seq]

            auto[tgtInput,tgtOutput]  = CreateDecoderInputOutput(item.target);
       
            auto tgtOut = model->forward(item.data, tgtInput);

            auto output = tgtOut.reshape({ -1, tgt_vocab_size });
            optimizer.zero_grad();
            auto tgt = tgtOutput.squeeze(0);
     
            auto loss = loss_fn(output, tgt);
            total_loss += loss.item<float>();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
           
            optimizer.step();

        }


        if (i % 10 == 0 || (i + 1 == max_train))
        {
            std::cout << "i: " << i + 1 << " , loss: " << total_loss << std::endl;
        }

        if (total_loss <= accuracy)
        {
            std::cout << "break... " << ", total_loss: " << total_loss << " , i: " << i + 1 << std::endl;
            break;
        }
    }
    std::cout << std::endl;
}

void TestData(Translator& model)
{
    model->eval();
    std::cout << "测试&翻译:" << std::endl;
    std::vector<std::string> tests;
   
    tests.push_back("Welcome to PyTorch Tutorials Pad Pad Pad Pad Pad");
    tests.push_back("Welcome to Machine Learning  Pad Pad Pad Pad");
    

    for (auto ch : tests)
    {
        auto item = GetWordId(src_vocab,ch);
        auto src = torch::tensor(item, torch::kLong);

        auto result = model->predict(src);
        
       // std::cout << std::regex_replace(ch, std::regex("Pad"), "") << " :  ";
         std::cout << ch << " :  ";


        for (int k = 0; k < result.numel(); k++)
        {
            std::cout << GetWordById(tgt_vocab,result[k].item<int64_t>()) << " ";
        }

        std::cout << std::endl;
    }

}
