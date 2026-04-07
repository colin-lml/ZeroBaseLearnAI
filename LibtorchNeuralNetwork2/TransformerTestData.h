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

typedef const std::unordered_map< std::string, int64_t> TableVocab;
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

