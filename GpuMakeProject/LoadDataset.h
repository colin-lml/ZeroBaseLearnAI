
#pragma once

#include <iostream>

#include <torch/torch.h>

#include <fstream>
#include <random>


using namespace std;


extern	int64_t  gBOS;
extern	int64_t  gEOS;
extern	int64_t  gPad;
extern  torch::DeviceType gDType;

vector<pair<vector<int64_t>, vector<int64_t>>> MakeTestData(const int count);


class translatDatasetOnly : public torch::data::Dataset<translatDatasetOnly>
{
public:

    translatDatasetOnly()
    {
        m_vTestData = MakeTestData(40);

    }
    torch::optional<size_t> size() const
    {
        return m_vTestData.size();
    }

    torch::data::Example<torch::Tensor, torch::Tensor>  get(size_t index) override
    {

        auto& item = m_vTestData.at(index);

        auto inpput = torch::tensor(item.first, torch::kLong);

        auto lable = torch::tensor(item.second, torch::kLong);

        return { inpput, lable };

    }

    vector<pair<vector<int64_t>, vector<int64_t>>> m_vTestData;
};



