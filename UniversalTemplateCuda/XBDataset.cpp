#include "pch.h"
#include "XBDataset.h"

static size_t gMaxBatch = 0;

static VectorTrainEncoded gTrainEncoded;

void SetVectorTrainEncoded(const VectorTrainEncoded& vte)
{
    gTrainEncoded = vte;
}

void  UpdateBatchMax(std::vector<size_t>& vIndex)
{
    gMaxBatch = 0;

    for (auto& v : vIndex)
    {
        gMaxBatch = max(gMaxBatch, gTrainEncoded.at(v).size());
    }

}



XBDataset::XBDataset() : m_vData(m_bbpe.GetTrainData())
{
     m_iBos = m_bbpe.GetBOS();
     m_iEos = m_bbpe.GetEOS();
     m_iPad = m_bbpe.GetPAD();

     SetVectorTrainEncoded(m_vData);
}

XBDataset::~XBDataset()
{
 
}

torch::optional<size_t> XBDataset::size() const
{
	return m_vData.size();
}

torch::data::Example<torch::Tensor, torch::Tensor>  XBDataset::get(size_t index)
{

    auto item = m_vData.at(index);
    int  len = gMaxBatch - item.size();
    item.insert(item.begin(), m_iBos);
    item.push_back(m_iEos);

    for (int i = 0; i < len; i++)
    {
        item.push_back(m_iPad);
    }

    auto inpput = torch::tensor(item, torch::kLong);

    item.erase(item.begin());
    item.push_back(m_iPad);

    auto lable = torch::tensor(item, torch::kLong);

    return { inpput, lable };

}



XBatchSampler::XBatchSampler(size_t size):RandomSampler(size)
{

}

std::optional<std::vector<size_t>> XBatchSampler::next(size_t batch_size)
{
    auto  vlist = RandomSampler::next(batch_size);
    if (vlist != std::nullopt)
    {
        UpdateBatchMax(*vlist);
    }
    return vlist;
}


