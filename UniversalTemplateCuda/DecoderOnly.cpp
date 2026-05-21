#include "pch.h"
#include "DecoderOnly.h"

const static int64_t gInt64Dim = 64;

const static bool gBoolBias = true;

XDecoderOnlyImpl::XDecoderOnlyImpl(int64_t numHeads, int64_t numWords, int64_t numLayer, int64_t pad)
{
    m_pad = pad;
    m_numLayers = numLayer;
    int64_t dim = gInt64Dim * numHeads;
    auto linear = torch::nn::LinearOptions(dim, numWords).bias(gBoolBias);

    m_fc = register_module("fc", torch::nn::Linear(linear));
    m_embPos = register_module("embPos", EmbeddingWithPosition(dim, numWords, m_pad));
    m_decoderLayers = register_module("decoderLayers", torch::nn::ModuleList());

    for (int i = 0; i < m_numLayers; i++)
    {
        m_decoderLayers->push_back(XDecoderLayer(dim, numHeads, dim * 4, gBoolBias));
    }
}

//// x [bath, seq]

torch::Tensor XDecoderOnlyImpl::forward(torch::Tensor x)
{
    
    if (x.dim() == 1)
    {
        x.unsqueeze_(0);
    }

    int64_t B = x.size(0);
    int64_t S = x.size(1);

    auto src_mask = generate_square_subsequent_mask(S).to(x.device());
   
    auto paddingMask = (x == m_pad).to(torch::kBool).to(x.device());

    x = m_embPos->forward(x);

    x = x.permute({ 1,0, 2 });

    for (auto& item : *m_decoderLayers)
    {
        x = item->as<XDecoderLayer>()->forward(x, src_mask, paddingMask);
    }

    return m_fc->forward(x);
}

torch::Tensor XDecoderOnlyImpl::generate_square_subsequent_mask(int64_t sz)
{
    auto mask = torch::triu(torch::ones({ sz, sz }, torch::kFloat32), 1);

    //mask = mask.masked_fill(mask == 1, -std::numeric_limits<float>::infinity());
    mask = mask.masked_fill(mask == 1, -1e9);
    return mask;  //
}

void XDecoderOnlyImpl::predict(torch::DeviceType xdevice, VectorInt64& input, int64_t eos, int64_t maxSeq)
{
    int start = input.size()-1;

    for (int i = 0; i < maxSeq; i++)
    {
        torch::Tensor tgt = torch::tensor(input, torch::kLong).to(xdevice);
        auto out = forward(tgt);
        out = out.squeeze();

        auto nextToken = out.argmax(-1).cpu();
        std::vector<int64_t> outVector;
        int64_t key = nextToken[start + i].item<int64_t>();
        input.push_back(key);

        if (key == eos)
        {
            break;
        }
    }
}

