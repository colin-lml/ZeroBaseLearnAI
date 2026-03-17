#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>

//#include <iostream>
#include <fstream>

#define  dim_model   32
#define  max_vocab_len  11
#define  max_train       100


const std::unordered_map< std::string, int64_t> maxVocabId =
{
    {"0",0}, 
    {"1",1}, 
    {"2",2},
    {"one",3},
    {"two",4},
    {"three",5},
    {"four",6},
    {"一",7},
    {"二",8},
    {"三",9},
    {"四",10}
};




typedef std::vector<std::pair<int64_t, int64_t>>  WordList;


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

std::string GetWordById(int64_t dataid)
{
    std::string Word = {"0"};
    for (auto& w : maxVocabId)
    {
        if (w.second == dataid)
        {
            Word = w.first;
            break;
        }  
    }
    return Word;
}

std::vector<int64_t> GetWordId(std::string data)
{
    std::vector<int64_t> input;
    for (auto ch : Split(data))
    {
        input.push_back(maxVocabId.at(ch));
    }
    return input;
}

WordList GetLoadDataWordId(std::pair<std::string, std::string> data)
{
    std::vector<int64_t> input = GetWordId(data.first);
    std::vector<int64_t> target = GetWordId(data.second);

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
        wordCount.push_back(GetLoadDataWordId({ "0","0" }));
        wordCount.push_back(GetLoadDataWordId({ "1","1" }));
        wordCount.push_back(GetLoadDataWordId({ "2","2" }));

        wordCount.push_back(GetLoadDataWordId({ "one","一" }));
        wordCount.push_back(GetLoadDataWordId({ "two","二" }));
        wordCount.push_back(GetLoadDataWordId({ "three","三" }));
        wordCount.push_back(GetLoadDataWordId({ "four","四" }));

         
        wordCount.push_back(GetLoadDataWordId({ "one two","一 二" }));
        wordCount.push_back(GetLoadDataWordId({ "two one","二 一" }));

        wordCount.push_back(GetLoadDataWordId({ "one two three ","一 二 三" }));
        wordCount.push_back(GetLoadDataWordId({ "three two one"," 三 二 一" }));

        wordCount.push_back(GetLoadDataWordId({ "one two three four","一 二 三 四" }));
        wordCount.push_back(GetLoadDataWordId({ "four three two one","四 三 二 一" }));
       

    }


    torch::optional<size_t> size() const
    {
        return wordCount.size();
    }

    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override
    {
        auto item = wordCount[index];
        std::vector<int64_t> tmpinput;
        std::vector<int64_t> tmptarget;

        for each(auto& i in item)
        {
            tmpinput.push_back(i.first);
            tmptarget.push_back(i.second);
        }

        auto input = torch::tensor(tmpinput, torch::kLong);
        auto target = torch::tensor(tmptarget, torch::kLong);

        return { input, target };
    }

public:
    std::vector<WordList> wordCount;
  
};


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
        _posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, _d_model, 2) },torch::sin(pos * den) );
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
        src_emb = register_module("src_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(max_vocab_len, dim_model)));
        tgt_emb_ = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(max_vocab_len, dim_model)));
        pos_encoder = register_module("pos_encoder", PositionalEncoding(dim_model, max_vocab_len));
        torch::nn::TransformerOptions opts;
        opts.nhead(1);
        opts.dim_feedforward(64);
        opts.num_decoder_layers(1);
        opts.num_encoder_layers(1);
        opts.dropout(0.0);
        opts.d_model(dim_model);
        transformer = register_module("transformer", torch::nn::Transformer(opts));
        fc = register_module("fc", torch::nn::Linear(dim_model, max_vocab_len));
    
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
        auto outs= transformer->forward(src, tgt);
   
        outs = fc->forward(outs);

        return outs;

    }
   
    torch::Tensor predict(torch::Tensor src) 
    {
        auto srcemb = src_emb->forward(src) * std::sqrt(dim_model);
        srcemb = pos_encoder->forward(srcemb);
        auto memory = transformer->encoder.forward(srcemb);

        std::vector<int64_t> tgtpad;
        tgtpad.push_back(0);
      
        for (int i=0; i < src.numel();i++)
        {
            torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong);
            auto tgt_emb = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
            tgt_emb = pos_encoder->forward(tgt_emb);
            auto out = transformer->decoder.forward(tgt_emb, memory);
            out = fc->forward(out).squeeze(-2);
            auto next_token = out.argmax(-1);
            tgtpad.insert(tgtpad.begin(), next_token[i].item<int64_t>());
        }
        tgtpad.pop_back();
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
    if (!bmodel || true)
    {
        TrainData(model);
        torch::save(model, model_path);
    }
    else
    {
        torch::load(model, model_path);
        std::cout<<"load model ...."<<std::endl;
    }
    filem.close();


    TestData(model);
   
}

void TrainData(Translator& model)
{
    double accuracy = 0.3;

    auto datasetTrain = translatDataset().map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(1));
    torch::nn::CrossEntropyLoss loss_fn;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();
    std::cout <<"训练模型" << std::endl;

    for (int i = 0; i < max_train; i++)
    {
        float total_loss = 0;
        for (auto& item : *train_data_loader)
        {
            /// item: [batch, seq]
            if (item.data.size(1) == 1 && item.data.item<int64_t>() <= 2)
            {
                continue;
            }

            auto tgtOut = model->forward(item.data, item.target);
            auto output = tgtOut.reshape({ -1, max_vocab_len });
            optimizer.zero_grad();
            auto tgt = item.target.squeeze(0);


            auto loss = loss_fn(output, tgt);
            total_loss += loss.item<float>();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();

        }


        if (i % 10 == 0 || (i + 1 == max_train))
        {
            std::cout << "i: " << i + 1 << " , loss: " << total_loss << std::endl ;
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
    tests.push_back("one");
    tests.push_back("two");
    tests.push_back("three");
    tests.push_back("four");
    /* 
    tests.push_back("one two");
    tests.push_back("two one");

    tests.push_back("one two three");
    tests.push_back("three two one");

    tests.push_back("one two three four");
    tests.push_back("four three two one");
    */
   
    for (auto ch: tests)
    {
        auto item = GetWordId(ch);
        auto src = torch::tensor(item, torch::kLong);
   
        auto result = model->predict(src);
        std::cout << ch <<" :  ";
        
        for (int  k = 0; k < result.numel(); k++) 
        {
            std::cout << GetWordById(result[k].item<int64_t>()) << " ";
        }

        std::cout << std::endl;
    }

}