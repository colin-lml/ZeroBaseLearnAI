
#pragma once

#include <iostream>

#include <torch/torch.h>

#include <fstream>
#include <random>

#include "LoadDataset.h"

using namespace std;

struct DeOnlyOptions
{
    int64_t dmodel = 512;
    int64_t head = 8;
    int64_t ffn = 2048;
    int64_t layers = 6;
    int64_t vocab_size = 3000;
    int64_t max_len = 5000;
};


class XMultiHeadAttentionImpl : public torch::nn::Module
{
public:

    XMultiHeadAttentionImpl(int64_t dim, int64_t head)
    {
        assert(dim % head == 0);

        InitQKV(dim, head);
    }

    //q k v : [seq, batch, dim]
    torch::Tensor forward(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, torch::Tensor& mask)
    {
        auto q1 = Q->forward(q);
        auto k1 = Q->forward(k);
        auto v1 = Q->forward(v);

        return ScaledDotProductAttention(q1, k1, v1, mask);
    }


private:
    void InitQKV(int64_t dim, int64_t head)
    {

        auto linear = torch::nn::LinearOptions(dim, dim).bias(false);

        Q = register_module("q", torch::nn::Linear(linear));
        K = register_module("k", torch::nn::Linear(linear));
        V = register_module("v", torch::nn::Linear(linear));
        Wo = register_module("Wo", torch::nn::Linear(linear)); // 输出投影

        norm_fact = 1.0 / sqrt(dim);

        Dk = dim / head;
        H = head;
    }

    /// q: [batch,seq, dim]
    torch::Tensor ScaledDotProductAttention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor& mask)
    {
        /// k==v q可以不等于 k v
        auto batch  = q.size(0);
        auto seq    =  q.size(1);
        auto dim    =  q.size(2);

        auto batch2 = k.size(0);
        auto seq2  = k.size(1);
        auto dim2 = k.size(2);

        q = q.view({ batch ,seq,  H, Dk }); // [b, s, h, d]
        k = k.view({ batch2,seq2, H, Dk });
        v = v.view({ batch2,seq2, H, Dk });

        q = q.permute({ 0,2,1,3 }); //[b, s, h, d] --->[B, H, S, Dk]
        k = k.permute({ 0,2,1,3 });
        v = v.permute({ 0,2,1,3 });

        auto kt = k.permute({ 0,1,3,2 }); //kt:  [B, H, S, Dk] --> [B, H, Dk, S]

        //cout << "q\n" << q.sizes() << endl;
        //cout << "kt\n" << kt.sizes() << endl;


        auto attn_score = torch::matmul(q, kt);

        attn_score = attn_score * norm_fact;

        if (mask.defined())
        {
            ///std::cout << "attn_score\n" << attn_score.sizes() << std::endl;
            //std::cout << "mask\n" << mask.sizes() << std::endl;
            attn_score += mask;
        }

        attn_score = torch::softmax(attn_score, -1); /// attn_score: [B, H, S, S]

        auto out = torch::matmul(attn_score, v); // [B, H, S, S] * [B, H, S, Dk]  ->  out: [B, H, S, Dk]
        out = out.transpose(1, 2).contiguous().view({ batch,seq, dim }); //  [B, H, S, Dk] --> [B, S, H, Dk] -> [batch,seq, dim]
        //cout <<"out\n" << out.squeeze() << endl;
        out = Wo->forward(out);
        return out;
    }

    torch::nn::Linear Q{ nullptr };
    torch::nn::Linear K{ nullptr };
    torch::nn::Linear V{ nullptr };
    torch::nn::Linear Wo{ nullptr };

    double norm_fact = 0;
    int64_t Dk;
    int64_t H;
};
TORCH_MODULE(XMultiHeadAttention);



class EmbeddingWithPositionImpl : public torch::nn::Module
{
public:
    EmbeddingWithPositionImpl(int64_t d_model, int64_t vocab_size, int64_t max_len)
    {
        m_iDmodel = d_model;
        m_iMaxlen = max_len;
        m_posEncode = torch::zeros({ m_iMaxlen, m_iDmodel }, torch::kFloat32);

        Encoding();
        m_emb = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, m_iDmodel)));
        register_buffer("posEncode", m_posEncode);
    }


    torch::Tensor& forward(torch::Tensor& x)
    {
        // x [bath, seq]
        //cout << x.sizes()<<endl;
        //cout <<m_emb << endl;

        x = m_emb->forward(x) * std::sqrt(m_iDmodel);

        // x [bath, seq, dim]

        auto dim = x.size(1);
        // cout << "x  " << x.sizes() << endl;
         //cout << "m_posEncode  "  << m_posEncode.sizes() << endl;
         //cout << "m_posEncode.slice  " << m_posEncode.slice(1, 0, dim).sizes() << endl;

        x = x + m_posEncode.slice(1, 0, dim);

        return  x;
    }

private:
    void Encoding()
    {
        auto pos = torch::arange(0, m_iMaxlen, torch::kFloat32).reshape({ m_iMaxlen, 1 });
        auto den_indices = torch::arange(0, m_iDmodel, 2, torch::kFloat32);
        auto den = torch::exp(-den_indices * std::log(10000.0f) / m_iDmodel);
        m_posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, m_iDmodel, 2) }, torch::sin(pos * den));
        m_posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, m_iDmodel, 2) }, torch::cos(pos * den));
        m_posEncode.unsqueeze_(0);
        // [bath,seq,dim]

    }

    torch::nn::Embedding m_emb{ nullptr };
    torch::Tensor m_posEncode;
    int64_t m_iDmodel = 0;
    int64_t m_iMaxlen = 0;
};

TORCH_MODULE(EmbeddingWithPosition);



class DeOnlyLayerImpl : public torch::nn::Module
{
public:
    DeOnlyLayerImpl(int64_t dmodel, int64_t nheads,int64_t feedforward)
    {
        torch::nn::LayerNormOptions normOpt({ dmodel });

        m_fFeedForward = register_module("feedForward", torch::nn::Sequential(torch::nn::Linear(dmodel, feedforward),
                         torch::nn::GELU(), torch::nn::Linear(feedforward, dmodel)));

       // torch::nn::MultiheadAttentionOptions opt;
        m_attention = register_module("multiAttention", XMultiHeadAttention(dmodel, nheads));
        
        m_norm1 = register_module("norm1", torch::nn::LayerNorm(normOpt));
        m_norm2 = register_module("norm2", torch::nn::LayerNorm(normOpt));

    }

    torch::Tensor forward(torch::Tensor& x, torch::Tensor& mask)
    {

        //auto kkk = m_attention->forward(q, k, v);

        //cout << x.sizes()<< endl;
        //cout << mask.sizes() << endl;

       // auto [attnOutput, attnWeights] = m_attention->forward(q, k, v,{},true, mask);
        auto attnOutput= m_attention->forward(x, x, x, mask);

        auto y = m_norm1->forward(attnOutput + x);
        auto y2 = m_fFeedForward->forward(y);

        return m_norm2->forward(y2 + y);
    }


    torch::nn::LayerNorm m_norm1{ nullptr }, m_norm2{ nullptr };
    //torch::nn::MultiheadAttention m_attention{ nullptr };
    XMultiHeadAttention m_attention{ nullptr };
    torch::nn::Sequential m_fFeedForward{ nullptr };
};
TORCH_MODULE(DeOnlyLayer);


class DecodersOnlyImpl : public torch::nn::Module
{

public:

    DecodersOnlyImpl(const DeOnlyOptions& inOpt)
    {
        fc = register_module("fc", torch::nn::Linear(inOpt.dmodel, inOpt.vocab_size));

        m_emb = register_module("m_emb", EmbeddingWithPosition(inOpt.dmodel, inOpt.vocab_size, inOpt.max_len));

        moduleLayers = register_module("EncoderLayers", torch::nn::ModuleList());

        m_option = inOpt;

        for (int i = 0; i < inOpt.layers; i++)
        {
            torch::nn::TransformerEncoderLayerOptions opt(inOpt.dmodel, inOpt.head);
            opt.dim_feedforward(inOpt.ffn);
            //opt.dropout(0);

            moduleLayers->push_back(DeOnlyLayer(inOpt.dmodel, inOpt.head, inOpt.ffn));
        }


    }

    auto forward(torch::Tensor x)
    {
        // x [bath, seq]
        int64_t batch = x.size(0);
        int64_t seq = x.size(1);
        auto src_mask = generate_square_subsequent_mask(seq).to(x.device());

        auto tgt_key_padding_mask = (x == gPad).to(torch::kBool).to(x.device());

        x = m_emb->forward(x);

       /// x = x.permute({ 1,0, 2 });
        // cout <<x.sizes()<< endl;



        for (auto& item : *moduleLayers)
        {
            x = item->as<DeOnlyLayer>()->forward(x, src_mask);

        }

        return fc->forward(x);
    }

    string predict(string ch, translatDatasetOnly& dataTest)
    {
        ch = "<BOS>" + ch;

        auto tgtpad = dataTest.GetTangshiCode(ch);


        int i = 0;
        while (i < 100)
        {
            torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong).to(gDType);
            auto out = forward(tgt.unsqueeze(0));
            
            out = out.squeeze();
           
            auto next_token = out.argmax(-1).cpu();
           // cout << next_token << endl;

            int64_t key = next_token[i].item<int64_t>();
            tgtpad.push_back(key);
            if (key == gEOS)
            {
                break;
            }
            i++;
        }

        return dataTest.GetTangshiString(tgtpad);
    }





    torch::Tensor generate_square_subsequent_mask(int64_t sz)
    {
        auto mask = torch::triu(torch::ones({ sz, sz }, torch::kFloat32), 1);

        mask = mask.masked_fill(mask == 1, -std::numeric_limits<float>::infinity());
        return mask;
    }


public:
    EmbeddingWithPosition m_emb{ nullptr };
    torch::nn::ModuleList moduleLayers{ nullptr };
    torch::nn::Linear fc{ nullptr };
    DeOnlyOptions m_option;
};
TORCH_MODULE(DecodersOnly);




