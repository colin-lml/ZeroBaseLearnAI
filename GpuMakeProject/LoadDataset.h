
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

vector<vector<int64_t>> MakeTestData(int count);

extern size_t m_gMaxBatch;

class translatDatasetOnly : public torch::data::Dataset<translatDatasetOnly>
{
public:

    translatDatasetOnly()
    {
#ifdef __TestData__

        m_vTestData = MakeTestData(40);
        
#else 

        m_dataToken.InitLoadDataSrc();
        m_vdata = m_dataToken.GetEncodeData();
        gVocabCount = m_dataToken.GetCorpusVocabCount();
        
        gPad = m_dataToken.GetPAD();
        gBOS = m_dataToken.GetBOS();
        gEOS = m_dataToken.GetEOS();;
        

#endif // DEBUG
    }
    torch::optional<size_t> size() const
    {
#ifdef __TestData__
        return m_vTestData.size();
#else

        return m_vdata.size();
#endif
    }

    torch::data::Example<torch::Tensor, torch::Tensor>  get(size_t index) override
    {

#ifdef __TestData__
  
        auto item  = m_vTestData.at(index);

#else
        auto item = m_vdata.at(index).content;
#endif
        int  len = m_gMaxBatch - item.size();
        item.insert(item.begin(), gBOS);
        item.push_back(gEOS);

        for (int i = 0; i < len; i++)
        {
            item.push_back(gPad);
        }

        auto inpput = torch::tensor(item, torch::kLong);

        item.erase(item.begin());
        item.push_back(gPad);

        auto lable = torch::tensor(item, torch::kLong);

        return { inpput, lable };

    }

    void UpdateBatchMax(std::vector<size_t>& vlist)
    {
        m_gMaxBatch = 0;
#ifdef __TestData__

        for (size_t i = 0; i < vlist.size(); i++)
        {
            m_gMaxBatch = max(m_gMaxBatch, m_vTestData.at(vlist.at(i)).size());
        }

        
#else

        for (size_t i = 0; i < vlist.size(); i++)
        {
            m_gMaxBatch = max(m_gMaxBatch, m_vdata.at(vlist.at(i)).content.size());
        }

#endif
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

    vector<vector<int64_t>> m_vTestData;
    
};

struct BatchSampler : public torch::data::samplers::RandomSampler
{
    BatchSampler(translatDatasetOnly* dataset) : RandomSampler(*dataset->size())
    {
        m_dataset = dataset;
    }

    std::optional<std::vector<size_t>> next(size_t batch_size) override
    {
        auto  vlist = RandomSampler::next(batch_size);
        if (vlist != std::nullopt)
        {
            m_dataset->UpdateBatchMax(*vlist);
        }  
        return vlist;
    }

    translatDatasetOnly* m_dataset;
};




