
#pragma once

#include <iostream>

#include <torch/torch.h>

#include <fstream>
#include <random>

#include "LoadDataset.h"

using namespace std;


#define dropoutp 0.0

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
        auto k1 = K->forward(k);
        auto v1 = V->forward(v);

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

        //attn_dropout = register_module("attn_dropout", torch::nn::Dropout(dropoutp));

        norm_fact = 1.0 / sqrt(dim);

        Dk = dim / head;
        H = head;
    }

    /// q: [seq,batch, dim]
    torch::Tensor ScaledDotProductAttention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor& mask)
    {
        /// k==v q可以不等于 k v
        auto batch  = q.size(1);
        auto seq    =  q.size(0);
        auto dim    =  q.size(2);

        auto batch2 = k.size(1);
        auto seq2  = k.size(0);
        auto dim2 = k.size(2);

        q = q.view({ seq, batch,  H, Dk }); // [s, b,  h, d]
        k = k.view({ seq2, batch2, H, Dk });
        v = v.view({ seq2, batch2, H, Dk });

        q = q.permute({ 1,2,0,3 }); //[s, b, h, d] --->[B, H, S, Dk]
        k = k.permute({ 1,2,0,3 });
        v = v.permute({ 1,2,0,3 });

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

        //attn_score = attn_dropout->forward(attn_score);

        torch::Tensor max_scores = std::get<0>(attn_score.max(-1, true));

        // 2. 数值稳定
        attn_score = attn_score - max_scores;

        // 3. 防止整行被 mask 导致 softmax 变成 NaN
        torch::Tensor all_masked = (max_scores == -1e9);
        attn_score = torch::where(all_masked, torch::zeros_like(attn_score), attn_score);


        attn_score = torch::softmax(attn_score, -1); /// attn_score: [B, H, S, S]
        //if (attn_score.sum().item<float>() < -1e8)
        {
            //cout << "[3] softmax sum: " << attn_score.sum().item<float>() << endl;
        }
       //

        auto out = torch::matmul(attn_score, v); // [B, H, S, S] * [B, H, S, Dk]  ->  out: [B, H, S, Dk]
        out = out.transpose(1, 2).contiguous().view({ seq,batch, dim }); //  [B, H, S, Dk] --> [B, S, H, Dk] -> [batch,seq, dim]
        //cout <<"out\n" << out.squeeze() << endl;
        out = Wo->forward(out);
        return out;
    }

    torch::nn::Linear Q{ nullptr };
    torch::nn::Linear K{ nullptr };
    torch::nn::Linear V{ nullptr };
    torch::nn::Linear Wo{ nullptr };

   // torch::nn::Dropout attn_dropout{ nullptr };

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
        emb_dropout = register_module("emb_dropout1", torch::nn::Dropout(dropoutp));
        m_embPosition = register_module("m_embPosition", torch::nn::Embedding(torch::nn::EmbeddingOptions(max_len, m_iDmodel)));
        m_emb = register_module("m_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, m_iDmodel)));
        register_buffer("posEncode", m_posEncode);
    }


    torch::Tensor& forward(torch::Tensor& x)
    {
        // x [bath, seq]
        //cout << x.sizes()<<endl;
        //cout <<m_emb << endl;

        x = m_emb->forward(x) * std::sqrt(m_iDmodel);
       
       // cout << x.sizes() << endl;

        // x [bath, seq, dim]

        auto seq = x.size(1);
        auto B = x.size(0);
        // cout << "x  " << x.sizes() << endl;
         //cout << "m_posEncode  "  << m_posEncode.sizes() << endl;
         

        auto pos = torch::arange(seq, x.device()).unsqueeze(0).expand({ B, seq });
       
       // cout << "x  " << x.sizes() << endl;
        //cout << "m_posEncode.slice  " << m_posEncode.slice(1, 0, seq).sizes() << endl;

         x = x + m_posEncode.slice(1, 0, seq);

    
         //x = x + m_embPosition->forward(pos);

         x = emb_dropout->forward(x);

        return  x ;
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
    torch::nn::Dropout emb_dropout{ nullptr };
    torch::nn::Embedding m_emb{ nullptr };
    torch::nn::Embedding m_embPosition{ nullptr };
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
        m_dropoutAtt = register_module("m_dropoutAtt", torch::nn::Dropout(dropoutp));
        m_dropoutFfn = register_module("m_dropoutFfn", torch::nn::Dropout(dropoutp));

    }

    torch::Tensor forward(torch::Tensor& x, torch::Tensor& mask)
    {

       // auto [attnOutput, attnWeights] = m_attention->forward(q, k, v,{},true, mask);
        auto attnOutput= m_attention->forward(x, x, x, mask);

        attnOutput = m_dropoutAtt->forward(attnOutput);
       
        auto y = m_norm1->forward(attnOutput + x);
        auto y2 = m_fFeedForward->forward(y);
        y2 = m_dropoutFfn->forward(y2);
       

        return m_norm2->forward(y2 + y);

    }


    torch::nn::LayerNorm m_norm1{ nullptr }, m_norm2{ nullptr };
    //torch::nn::MultiheadAttention m_attention{ nullptr };
    XMultiHeadAttention m_attention{ nullptr };
    torch::nn::Sequential m_fFeedForward{ nullptr };
    torch::nn::Dropout m_dropoutAtt{ nullptr };
    torch::nn::Dropout m_dropoutFfn{ nullptr };
   
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

           /// moduleLayers->push_back(torch::nn::TransformerEncoderLayer(opt));
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

        x = x.permute({ 1,0, 2 });
        // cout <<x.sizes()<< endl;



        for (auto& item : *moduleLayers)
        {
             x = item->as<DeOnlyLayer>()->forward(x, src_mask);
            //x = item->as<torch::nn::TransformerEncoderLayer>()->forward(x, src_mask, tgt_key_padding_mask);
            
        }

        return fc->forward(x);
    }

#ifdef __TestData__

    string predict(string ch, translatDatasetOnly& dataTest)
    {
        std::vector<int64_t> tgtpad;

        tgtpad.push_back(gBOS);
        tgtpad.push_back(5);
        tgtpad.push_back(6);
        tgtpad.push_back(7);
        int start = tgtpad.size() - 1;
        std::vector<int64_t> outVector;
        int i = 0;
        while (i < 100)
        {
            torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong).to(gDType);
            auto out = forward(tgt.unsqueeze(0));

           // out = out.squeeze();

           //cout << out.sizes() << endl;
            auto next_token = out.argmax(-1).cpu();
           // cout << next_token << endl;
            
            int64_t key = next_token[i+ start].item<int64_t>();
            
           
            tgtpad.push_back(key);
            if (key == gEOS)
            {
                break;
            }
            i++;
            
        }
        cout  << "tgtpad ";
        for (auto& key : tgtpad)
        {
            cout << key << " ";
        }

        cout << endl << endl<<endl;
        

        return "__TestData__";
    }
#else

    string predict(string ch, translatDatasetOnly& dataTest)
    {
        ch = BOS + ch;

        auto tgtpad = dataTest.GetTangshiCode(ch);
        int start = tgtpad.size() - 1;

        int i = 0;
        while (i < 50)
        {
            torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong).to(gDType);
            auto out = forward(tgt.unsqueeze(0));
            
            out = out.squeeze();
           
            //cout << out << endl;
            auto next_token = out.argmax(-1).cpu();
            //cout << next_token << endl;
            std::vector<int64_t> outVector;
            int64_t key = next_token[i+start].item<int64_t>();

            //cout << dataTest.GetTangshiString(outVector) << endl;
            
            tgtpad.push_back(key);
            if (key == gEOS)
            {
                break;
            }
            i++;
           // cout<< dataTest.GetTangshiString(tgtpad) <<endl;

        }

        return dataTest.GetTangshiString(tgtpad);
    }
#endif




    torch::Tensor generate_square_subsequent_mask(int64_t sz)
    {
        auto mask = torch::triu(torch::ones({ sz, sz }, torch::kFloat32), 1);

        //mask = mask.masked_fill(mask == 1, -std::numeric_limits<float>::infinity());
        mask = mask.masked_fill(mask == 1, -1e9);
        return mask;  //
    }


public:
    EmbeddingWithPosition m_emb{ nullptr };
    torch::nn::ModuleList moduleLayers{ nullptr };
    torch::nn::Linear fc{ nullptr };
    DeOnlyOptions m_option;
};
TORCH_MODULE(DecodersOnly);




