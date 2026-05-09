
#pragma once

#include <iostream>

#include <torch/torch.h>

#include <fstream>
#include <random>
#include "Tokenizer.h"

using namespace std;


extern	int64_t  gBOS;
extern	int64_t  gEOS;
extern	int64_t  gPad;
extern	int64_t gVocabCount;
extern  torch::DeviceType gDType;

vector<pair<vector<int64_t>, vector<int64_t>>> MakeTestData(const int count);


class translatDatasetOnly : public torch::data::Dataset<translatDatasetOnly>
{
public:

    translatDatasetOnly()
    {
        m_dataToken.InitLoadDataSrc();
        m_vdata = m_dataToken.GetEncodeData();
        gVocabCount = m_dataToken.GetCorpusVocabCount();
        
        gPad = m_dataToken.GetPAD();
        gBOS = m_dataToken.GetBOS();
        gEOS = m_dataToken.GetEOS();;
       // m_vTestData = MakeTestData(40);

        for (auto& item : m_vdata)
        {
            m_nMaxTitle = max(m_nMaxTitle, item.title.size());
            m_nMaxAuthor = max(m_nMaxAuthor, item.author.size());
            m_nMaxContent = max(m_nMaxContent, item.content.size());
        }

        for (auto& item : m_vdata)
        {
            VectorCodeID input;
            input.push_back(gBOS);
            auto addPadLen = m_nMaxContent - item.content.size();
            input.insert(input.end(), item.content.begin(), item.content.end());
            input.push_back(gEOS);
            for (size_t i = 0; i < addPadLen; i++)
            {
                input.push_back(gPad);
            }
            InputContent.push_back(input);
            input.erase(input.begin());
            input.push_back(gPad);
            LableContent.push_back(input);
        }


    }
    torch::optional<size_t> size() const
    {
        return InputContent.size();
    }

    torch::data::Example<torch::Tensor, torch::Tensor>  get(size_t index) override
    {

        auto inpput = torch::tensor(InputContent.at(index), torch::kLong);

        auto lable = torch::tensor(LableContent.at(index), torch::kLong);

        return { inpput, lable };

    }

    std::vector<int64_t> GetTangshiCode(std::string& line)
    {
        return m_dataToken.Encode(line);
    }
    std::string GetTangshiString(std::vector<int64_t>& vList)
    {
        return m_dataToken.Decode(vList);
    }

    std::vector<VectorCodeTangshi> m_vdata;
    Tokenizer m_dataToken;
    size_t m_nMaxTitle = 0;
    size_t m_nMaxAuthor = 0;
    size_t m_nMaxContent = 0;

    std::vector<VectorCodeID> InputContent;
    std::vector <VectorCodeID> LableContent;
   // vector<pair<vector<int64_t>, vector<int64_t>>> m_vTestData;
};



