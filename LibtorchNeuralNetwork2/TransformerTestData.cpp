#include "TransformerTestData.h"

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

std::string GetWordById(TableVocab& vocabId, int64_t dataid)
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


std::vector<int64_t> GetWordId(TableVocab& vocabId, std::string data)
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
    std::vector<int64_t> input = GetWordId(src_vocab, data.first);
    std::vector<int64_t> target = GetWordId(tgt_vocab, data.second);

    WordList item;
    for (int i = 0; i < input.size() && i < target.size(); i++)
    {
        item.push_back({ input.at(i),target.at(i) });
    }

    return item;
}

std::pair<torch::Tensor, torch::Tensor>  CreateDecoderInputOutput(torch::Tensor data)
{
    auto E = torch::tensor(GetWordId(tgt_vocab, "E"), torch::kLong).view({ 1,1 });
    auto S = torch::tensor(GetWordId(tgt_vocab, "S"), torch::kLong).view({ 1,1 });

    auto  input = torch::cat({ S, data }, 1);
    auto  output = torch::cat({ data,E }, 1);

    return { input ,output };
}

