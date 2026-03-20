#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>

#include <fstream>


#if 1


#define  dim_model   240
#define  max_vocab_len  11
#define  max_train       50


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

std::vector<int64_t> GetWordId(std::string data, bool bAddflg = true)
{
 
    std::vector<int64_t> input;

    if (bAddflg)
    {
        input.push_back(1);
    }

    for (auto ch : Split(data))
    {
        input.push_back(maxVocabId.at(ch));
    }

    if (bAddflg)
    {
        input.push_back(2);
    }

    return input;
}

WordList GetLoadDataWordId(std::pair<std::string, std::string> data,bool bAddflg=true)
{
    std::vector<int64_t> input = GetWordId(data.first, bAddflg);
    std::vector<int64_t> target = GetWordId(data.second, bAddflg);

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
        wordCount.push_back(GetLoadDataWordId({ "0","0" },false));
        wordCount.push_back(GetLoadDataWordId({ "1","1" }, false));
        wordCount.push_back(GetLoadDataWordId({ "2","2" }, false));
         
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

        tmpinput.push_back(1);
        tmptarget.push_back(1);

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
        opts.nhead(6);
        opts.dim_feedforward(1024);
        opts.num_decoder_layers(3);
        opts.num_encoder_layers(3);
        opts.dropout(0.0);
        opts.d_model(dim_model);
        transformer = register_module("transformer", torch::nn::Transformer(opts));
        fc = register_module("fc", torch::nn::Linear(dim_model, max_vocab_len));
    
    }

    torch::Tensor generate_square_subsequent_mask(int64_t len)
    {
         auto mask = torch::ones({ len ,len });
         mask = torch::triu(mask,1);
         mask = mask.masked_fill(mask == 1, -1e9);
         return mask;
    }

    torch::Tensor create_padding_mask(torch::Tensor src, int64_t pad_idx = 0)
    {
        auto mask = (src == pad_idx);
        //std::cout << mask << std::endl;
        return mask;
    }

    std::pair<torch::Tensor, torch::Tensor> create_mask(torch::Tensor src, torch::Tensor tgt)
    {
       int64_t srcLen =  src.size(0);
       int64_t tgtLen =  tgt.size(0);
      
       auto tgtMask = generate_square_subsequent_mask(srcLen);

       auto srcMask = torch::zeros({ srcLen, srcLen }, torch::kBool);

       return { srcMask,tgtMask };
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt)
    {
       auto src_key_padmask = create_padding_mask(src); // key_padding_mask: [batch, seq]
       auto tgt_key_padmask = create_padding_mask(tgt);  //key_padding_mask:  [batch, seq]

        //[batch, seq]  --> [seq, batch]
        src = src.permute({ 1,0 });
        tgt = tgt.permute({ 1,0 });

       auto[src_mask, tgt_mask] = create_mask(src, tgt);

    
        //std::cout << "input " << src << std::endl;
        src = src_emb->forward(src) * std::sqrt(dim_model);
        src = pos_encoder->forward(src);

        tgt = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
        tgt = pos_encoder->forward(tgt);
        
        // tgt & src: (seq, batch, dim)
        auto outs = transformer->forward(src, tgt, src_mask, tgt_mask, torch::Tensor(), src_key_padmask, tgt_key_padmask);
   
        outs = fc->forward(outs);

        return outs;

    }
   
    torch::Tensor predict(torch::Tensor src) 
    {
        auto srcLen = src.size(0);

        auto srcMask = torch::zeros({ srcLen, srcLen }, torch::kBool);

        auto srcemb = src_emb->forward(src) * std::sqrt(dim_model);
        srcemb = pos_encoder->forward(srcemb);
        auto memory = transformer->encoder.forward(srcemb, srcMask);

        std::vector<int64_t> tgtpad;
        tgtpad.push_back(0);
          
        for (int i=0; i < src.numel();i++)
        {
            torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong);
            auto tgt_mask = generate_square_subsequent_mask(tgt.size(0));
           
            auto tgt_emb = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
            tgt_emb = pos_encoder->forward(tgt_emb);
            auto out = transformer->decoder.forward(tgt_emb, memory, tgt_mask);
            out = fc->forward(out).squeeze(-2);
            auto next_token = out.argmax(-1);
            tgtpad.insert(tgtpad.begin()+ tgtpad.size()-1, next_token[i].item<int64_t>());
            //tgtpad.push_back(next_token[i].item<int64_t>());
        }

        tgtpad.erase(tgtpad.begin());
        ///tgtpad.pop_back();
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
    double accuracy = 0.09;

    auto datasetTrain = translatDataset().map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(1));
    torch::nn::CrossEntropyLoss loss_fn{ torch::nn::CrossEntropyLossOptions().reduction(torch::kNone).ignore_index(0)};
   
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();
    std::cout <<"训练模型" << std::endl;

    for (int i = 0; i < max_train; i++)
    {
        float total_loss = 0;
        for (auto& item : *train_data_loader)
        {
            /// item: [batch, seq]
            //if (item.data.size(1) == 1 && item.data.item<int64_t>() <= 2)
            //{
             //   continue;
            //}

            auto tgtOut = model->forward(item.data, item.target);
            auto output = tgtOut.reshape({ -1, max_vocab_len });
            optimizer.zero_grad();
            auto tgt = item.target.squeeze(0);

            auto loss = loss_fn(output, tgt);
        
            auto mask = (tgt != 0).to(torch::kFloat);
            //std::cout << mask << std::endl;
            loss = (loss * mask).sum() / mask.sum();
            ///std::cout << loss << std::endl;
            total_loss += loss.item<float>();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();

        }


        if (i % 3 == 0 || (i + 1 == max_train))
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

    
    tests.push_back("one two");
    tests.push_back("two one");

    tests.push_back("one two three");
    tests.push_back("three two one");
    

    tests.push_back("one two three four");

    tests.push_back("four three two one");
    

  
   
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

#else

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

using namespace torch;
using namespace std;

// ========================= 工具函数 =========================
vector<string> split(const string& s) 
{
    vector<string> res;
    stringstream ss(s);
    string word;
    while (ss >> word) res.push_back(word);
    return res;
}

// 位置编码
Tensor positional_encoding(int64_t seq_len, int64_t d_model) 
{
    Tensor pe = torch::zeros({ seq_len, d_model });
    Tensor pos = torch::arange(0, seq_len, torch::kFloat).unsqueeze(1);
    Tensor div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat) * (-log(10000.0) / d_model));

    pe.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, d_model, 2) }, torch::sin(pos * div_term));
    pe.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, d_model, 2) }, torch::cos(pos * div_term));
    //_posEncode.unsqueeze_(-2);
    return pe;
}

// 生成未来词 Mask (Look-ahead Mask)
Tensor generate_square_mask(int64_t sz) 
{
    auto mask = torch::triu(torch::ones({ sz, sz }, torch::kFloat), 1);
    // 把 1 变成 -inf，让注意力变成极小（完全屏蔽）
    mask = mask.masked_fill(mask == 1, -1e9);
    return mask;
}

// 生成 Padding Mask
Tensor create_pad_mask(const Tensor& seq, int64_t pad_idx = 0) 
{
    return (seq == pad_idx).to(torch::kBool);
}

// ========================= 词表 =========================
unordered_map<string, int64_t> src_vocab = 
{
    {"<PAD>",0}, {"<SOS>",1}, {"<EOS>",2},
    {"1",3}, {"2",4}, {"3",5}
};

unordered_map<int64_t, string> tgt_vocab =
{
    {0,"<PAD>"}, {1,"<SOS>"}, {2,"<EOS>"},
    {3,"一"}, {4,"二"}, {5,"三"}
};

// ========================= Transformer 模型 =========================
struct TransformerDemo : nn::Module
{
public:
    int64_t d_model = 64;
    int64_t vocab_size = 6;

    nn::Embedding src_emb{ vocab_size, d_model };
    nn::Embedding tgt_emb{ vocab_size, d_model };

    nn::TransformerOptions option = nn::TransformerOptions(d_model, 2, 1, 1).dropout(0.0);
    
    nn::Transformer transformer{ option };

    nn::Linear fc{ d_model, vocab_size };

    TransformerDemo() {
        register_module("src_emb", src_emb);
        register_module("tgt_emb", tgt_emb);
        register_module("transformer", transformer);
        register_module("fc", fc);
    }

    Tensor forward(Tensor src, Tensor tgt)
    {
        int64_t src_seq = src.size(1);
        int64_t tgt_seq = tgt.size(1);

        // ====================== 生成 Mask ======================
        auto  src_pad_mask = create_pad_mask(src);// .transpose(1, 0);
        auto tgt_pad_mask = create_pad_mask(tgt);// .transpose(1, 0);              // [B,T]
        
        //auto tgt_sub_mask = generate_square_mask(tgt_seq).to(torch::kFloat);
        //tgt_sub_mask = tgt_sub_mask.masked_fill(tgt_pad_mask.unsqueeze(1) == 0, -1e9);
        //tgt_pad_mask = tgt_pad_mask.unsqueeze(1);

        // ====================== 嵌入 + 位置编码 ======================
        auto se = src_emb(src) + positional_encoding(src_seq, d_model);
        auto te = tgt_emb(tgt) + positional_encoding(tgt_seq, d_model);
        
        auto tgt_mask = generate_square_mask(tgt_seq);
        auto src_mask = torch::zeros({ src_seq, src_seq }, torch::kBool);

      
        //cout << "se1" << se.sizes() << endl;
        //cout << "te1" << te.sizes() << endl;

        se = se.permute({ 1,0,2 });
        te = te.permute({ 1,0,2 });

        //cout <<"src_pad_mask" << src_pad_mask << endl;
        //cout <<"tgt_pad_mask" << tgt_pad_mask << endl;


        Tensor memory_mask = {};
       // auto out = transformer->forward(se, te, src_mask, tgt_mask, memory_mask, src_pad_mask, tgt_pad_mask);
        auto out = transformer->forward(se, te, src_mask, tgt_mask, memory_mask, src_pad_mask, tgt_pad_mask, src_pad_mask);
       
       // auto out = transformer(se, te, src_pad_mask, tgt_sub_mask, src_pad_mask);
        
        out = fc(out);
        return out.permute({ 1,0,2 });
    }
};

// ========================= 翻译函数 =========================
string translate(shared_ptr<TransformerDemo>& model, const string& input) 
{
    vector<string> words = split(input);
    vector<int64_t> src_ids = { 1 };
    for (auto& w : words) src_ids.push_back(src_vocab[w]);
    src_ids.push_back(2);

    Tensor src = torch::tensor(src_ids, kLong).unsqueeze(0);
    vector<int64_t> tgt_ids = { 0 };

    for (int i = 0; i < 10; i++) 
    {
        Tensor tgt = torch::tensor(tgt_ids, kLong).unsqueeze(0);
        Tensor out = model->forward(src, tgt);
        cout << out.argmax(-1) << endl;
        Tensor next = out.argmax(-1);
        int64_t token = next[0][i].item<int64_t>();
        if (token == 2) break;
        tgt_ids.push_back(token);
    }

    string res;
    for (int i = 1; i < tgt_ids.size(); i++) res += tgt_vocab[tgt_ids[i]] + " ";
    return res;
}

// ========================= 主函数：训练 + 测试 =========================
 
torch::Tensor generate_square_mask2(int64_t size) 
{
    auto mask = (torch::triu(torch::ones({ size, size })) == 1).t();
    mask = mask.to(torch::kFloat);
    mask = mask.masked_fill(mask == 0, -1e9);
    mask = mask.masked_fill(mask == 1, 0.0);
    return mask;
}


void TransformerMain() 
{
    torch::Tensor a = generate_square_mask(3);
    torch::Tensor b = generate_square_mask2(3);

    cout << a << endl;
    cout << b << endl;
 
    // 1. 初始化
    auto model = make_shared<TransformerDemo>();
    nn::CrossEntropyLoss criterion{ nn::CrossEntropyLossOptions().reduction(torch::kNone) };
    optim::Adam optimizer(model->parameters(), optim::AdamOptions(1e-3));

    // 训练数据：1 2 3 → 一 二 三
    Tensor src = torch::tensor({ {1,3,4,5,2} }, kLong);
    Tensor tgt = torch::tensor({ {1,3,4,5,2} }, kLong);

    // 2. 训练
    cout << "开始训练..." << endl;
    for (int epoch = 0; epoch < 100; epoch++) {
        optimizer.zero_grad();
        Tensor out = model->forward(src, tgt);

        auto input = out.reshape({ -1, 6 });
        auto target = tgt.reshape({ -1 });
        auto loss = criterion(input, target);
        auto mask = (target != 0).to(kFloat);
        auto final_loss = (loss * mask).sum() / mask.sum();

        final_loss.backward();
        optimizer.step();

        if (epoch % 20 == 0)
            cout << "Epoch " << epoch << " | Loss: " << final_loss.item<float>() << endl;
    }

    // 3. 翻译测试
    cout << "\n===== 翻译结果 =====" << endl;
    cout << "输入：1 2 3" << endl;
    cout << "输出：" << translate(model, "1 2 3") << endl;

    
}




#endif



