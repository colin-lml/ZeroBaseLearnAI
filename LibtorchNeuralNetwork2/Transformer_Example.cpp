#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>

#include "TransformerTestData.h"

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
        auto none_mask = torch::Tensor();
        auto srclen =src.size(1);
        auto src_mask =torch::zeros({srclen,srclen}, torch::kBool);
        auto tgt_mask = transformer->generate_square_subsequent_mask(tgt.size(1));
        auto src_key_padding_mask = (src == PadId).to(torch::kBool);  // [batch,seq]
        auto tgt_key_padding_mask = (tgt == PadId).to(torch::kBool);  // [batch,seq]
        auto memory_key_padding_mask = src_key_padding_mask;
        //std::cout << "tgt_mask\n" << tgt_mask << std::endl;
        //std::cout << "src_key_padding_mask\n" << src_key_padding_mask << std::endl;
        //std::cout << "tgt_key_padding_mask\n" << tgt_key_padding_mask << std::endl;


        //[batch, seq]  --> [seq, batch]
        src = src.permute({ 1,0 });
        tgt = tgt.permute({ 1,0 });

        //std::cout << "input " << src << std::endl;
        src = src_emb->forward(src) * std::sqrt(dim_model);
        src = pos_encoder->forward(src);

        tgt = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
        tgt = pos_encoder->forward(tgt);

        // tgt & src: (seq, batch, dim)
        //auto outs = transformer->forward(src, tgt);
        auto outs = transformer->forward(src, tgt, src_mask, tgt_mask, none_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask);

        outs = fc->forward(outs);
        
        return outs;

    }


    torch::Tensor predict(torch::Tensor src)
    {
        //std::cout << src << std::endl;
        auto srclen = src.size(0);
        auto src_mask = torch::zeros({ srclen,srclen }, torch::kBool);
        auto srcemb = src_emb->forward(src) * std::sqrt(dim_model);
        srcemb = pos_encoder->forward(srcemb);
        auto memory = transformer->encoder.forward(srcemb, src_mask);

        std::vector<int64_t> tgtpad = GetWordId(tgt_vocab, "S");
     
        int i = 0;
        while (i < tgt_vocab_size*2)
        {
            torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong);
            auto tgt_mask = transformer->generate_square_subsequent_mask(tgt.size(0));
            ///std::cout << "tgt_mask " << tgt_mask << std::endl;
            auto tgt_emb = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
            tgt_emb = pos_encoder->forward(tgt_emb);
            auto out = transformer->decoder.forward(tgt_emb, memory, tgt_mask);
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


std::tuple<int64_t, int64_t, int64_t> count_model_parameters(Translator& model) 
{
    int64_t total_params = 0;
    int64_t trainable_params = 0;
    int64_t non_trainable_params = 0;

    
    for (const auto& p : model->parameters())
    {
 
        int64_t numel = p.numel();
        total_params += numel;
    
        if (p.requires_grad()) 
        {
            trainable_params += numel;
        }
        else 
        {
            non_trainable_params += numel;
        }
    }

    return { total_params, trainable_params, non_trainable_params };
}


void TransformerMain()
{
    torch::manual_seed(4);

    std::string model_path = "translator_model.pt";
    Translator model;

    auto[a,b,c] = count_model_parameters(model);

    std::cout << "模型总参数: "<< a << std::endl;
    std::cout << "可训练参数: " << b << std::endl;
    std::cout << "不可训参数: " << c << std::endl << std::endl;

    std::ifstream filem(model_path);
    bool bmodel = filem.is_open();
    if (!bmodel||true)
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
    auto options = torch::nn::CrossEntropyLossOptions().ignore_index(PadId);
    torch::nn::CrossEntropyLoss loss_fn(options);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();
    std::cout << "训练模型" << std::endl;

    for (int i = 0; i < max_train; i++)
    {
        float total_loss = 0;
        for (auto& item : *train_data_loader)
        {
            /// item.data, item.target : [batch, seq]

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
            std::cout <<"i: " << i + 1  << " , loss: " << total_loss <<  std::endl << std::endl;
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
    tests.push_back("Welcome");
    tests.push_back("Welcome to");
    tests.push_back("Welcome to PyTorch");
    tests.push_back("Welcome to Machine");
    tests.push_back("Welcome to PyTorch Tutorials");
    tests.push_back("Welcome to Machine Learning");

    tests.push_back("Learning");
    tests.push_back("Tutorials");
    tests.push_back("PyTorch Tutorials");
    tests.push_back("Machine Learning");

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
