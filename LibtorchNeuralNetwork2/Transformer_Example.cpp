#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>

//#include <iostream>
#include <fstream>

#define  dim_model   32
#define  max_vocab_len  11
#define  max_train       100

const std::unordered_map<int64_t, std::string> maxVocabId =
{
    {0, "0"}, {1, "1"}, {2, "2"},
    {3, "one"}, 
    {4, "two"}, 
    {5, "three"},
    {6, "four"},
    {7, "一"}, 
    {8, "二"}, 
    {9, "三"}, 
    {10, "四"}
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
        _posEncode.slice(0, 0, dim);
        //std::cout << _posEncode.slice(0, 0, dim) << std::endl;
        ///std::cout << x << std::endl;
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
        opts.dim_feedforward(32);
        opts.num_decoder_layers(1);
        opts.num_encoder_layers(1);
        opts.dropout(0.0);
        opts.d_model(dim_model);
        transformer = register_module("transformer", torch::nn::Transformer(opts));
        fc = register_module("fc", torch::nn::Linear(dim_model, max_vocab_len));
    
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt)
    {
  
        src = src_emb->forward(src) * std::sqrt(dim_model);
        src = pos_encoder->forward(src);

        tgt = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
        tgt = pos_encoder->forward(tgt);
        
        auto outs= transformer->forward(src, tgt);
   
        outs = fc->forward(outs);

        return outs;

    }
    torch::Tensor predict(torch::Tensor src)
    {
        src = src_emb->forward(src) * std::sqrt(dim_model);
        src = pos_encoder->forward(src);
        auto memory = transformer->encoder.forward(src);
       
        torch::Tensor tgt = torch::tensor({0}, torch::kLong);
     
        auto tgt_emb = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
        tgt_emb = pos_encoder->forward(tgt_emb);

        auto out = transformer->decoder.forward(tgt_emb, memory);
        out = fc->forward(out).squeeze(-2);
 
        auto next_token = out.argmax(-1);
        return next_token;

    }

    torch::nn::Embedding src_emb{ nullptr };
    torch::nn::Embedding tgt_emb_{ nullptr };
    PositionalEncoding pos_encoder{ nullptr };
    torch::nn::Transformer transformer{ nullptr };
    torch::nn::Linear fc{ nullptr };
};
TORCH_MODULE(Translator);

void TransformerMain() 
{
    torch::manual_seed(4);

    double accuracy = 0.05;

    torch::Tensor src_indices1 = torch::tensor({ 3 }, torch::kLong);
    torch::Tensor tgt_indices1 = torch::tensor({ 7 }, torch::kLong);

    torch::Tensor src_indices2 = torch::tensor({ 4 }, torch::kLong);
    torch::Tensor tgt_indices2 = torch::tensor({ 8 }, torch::kLong);

    torch::Tensor src_indices3 = torch::tensor({ 5 }, torch::kLong);
    torch::Tensor tgt_indices3 = torch::tensor({ 9 }, torch::kLong);

    torch::Tensor src_indices4 = torch::tensor({ 6 }, torch::kLong);
    torch::Tensor tgt_indices4 = torch::tensor({ 10 }, torch::kLong);

    torch::Tensor src_indices5 = torch::tensor({ 3, 4, 5, 6}, torch::kLong);
    torch::Tensor tgt_indices5 = torch::tensor({ 7, 8, 9, 10}, torch::kLong);
    std::string model_path = "translator_model.pt";

    Translator model;
   

    std::ifstream filem(model_path);
    bool bmodel = filem.is_open();
    if (!bmodel)
    {
        torch::nn::CrossEntropyLoss loss_fn;
        
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
        model->train();

        for (int i = 0; i < max_train; i++)
        {
  
            auto tgtOut = model->forward(src_indices1, tgt_indices1);
            auto output = tgtOut.reshape({ -1, max_vocab_len });
            optimizer.zero_grad();

            auto loss = loss_fn(output, tgt_indices1);

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();

        
            tgtOut = model->forward(src_indices1, tgt_indices1);
            output = tgtOut.reshape({ -1, max_vocab_len });
            optimizer.zero_grad();
            loss = loss_fn(output, tgt_indices1);
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();


          
            tgtOut = model->forward(src_indices2, tgt_indices2);
            output = tgtOut.reshape({ -1, max_vocab_len });
            optimizer.zero_grad();
            loss = loss_fn(output, tgt_indices2);
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();

           
            tgtOut = model->forward(src_indices3, tgt_indices3);
            output = tgtOut.reshape({ -1, max_vocab_len });
            optimizer.zero_grad();
            loss = loss_fn(output, tgt_indices3);
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();

            
            tgtOut = model->forward(src_indices4, tgt_indices4);
            output = tgtOut.reshape({ -1, max_vocab_len });
            optimizer.zero_grad();
            loss = loss_fn(output, tgt_indices4);
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();

            tgtOut = model->forward(src_indices5, tgt_indices5);
            output = tgtOut.reshape({ -1, max_vocab_len });
            optimizer.zero_grad();
            loss = loss_fn(output, tgt_indices5);
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();

            if (i % 10 == 0 || (i+1 == max_train))
            {
                std::cout << "i: " << i + 1 << " , loss: " << loss.item<double>() << std::endl;
            }

           // if (loss.item<double>() <= accuracy)
            {
               // std::cout << "break... " << ", loss: " << loss.item<double>() << " , i: " << i + 1 << std::endl;
                //break;
            }
        }
        torch::save(model, model_path);
    }
    else
    {
        torch::load(model, model_path);
        std::cout<<"load model ...."<<std::endl;
    }
    filem.close();

    model->eval();
    std::cout << "predict:" << std::endl;

    torch::Tensor src_test = torch::tensor({ 3 }, torch::kLong);
    auto result = model->predict(src_test);

    std::cout <<"3: " << result.item() << std::endl;

    src_test.fill_({4});
    result = model->predict(src_test);
    std::cout << "4: " << result.item() << std::endl;

    src_test.fill_({ 5 });
    result = model->predict(src_test);
    std::cout << "5: " << result.item() << std::endl;

    src_test.fill_({ 6 });
    result = model->predict(src_test);
    std::cout << "6: " << result.item() << std::endl;

}