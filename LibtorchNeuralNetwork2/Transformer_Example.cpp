#include <torch/torch.h>
#include <iostream>
#include <cmath>

// 位置编码层
class PositionalEncodingImpl : public torch::nn::Module 
{
public:
    PositionalEncodingImpl(int d_model, double dropout = 0.1, int max_len = 5000) 
    {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));

        // 计算位置编码矩阵
        pe = torch::zeros({ max_len, d_model });
        auto position = torch::arange(0, max_len, torch::kFloat).unsqueeze(1);
        auto div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat) *
            (-std::log(10000.0) / d_model));

        pe.slice(1, 0, d_model, 2) = torch::sin(position * div_term);
        pe.slice(1, 1, d_model, 2) = torch::cos(position * div_term);
        pe = pe.unsqueeze(0);  // [1, max_len, d_model]

        // 注册为缓冲区（不参与参数更新）
        register_buffer("pe", pe);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        // x: [batch_size, seq_len, d_model]
        x = x + pe.slice(1, 0, x.size(1));
        return dropout_->forward(x);
    }

private:
    torch::nn::Dropout dropout_;
    torch::Tensor pe;
};

TORCH_MODULE(PositionalEncoding);

// 缩放点积注意力
torch::Tensor scaled_dot_product_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,torch::optional<torch::Tensor> mask = torch::nullopt)
{

    int d_k = q.size(-1);
    // Q @ K^T / sqrt(d_k)
    auto attn_scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(d_k);

    // 应用掩码（填充/未来信息掩码）
    if (mask.has_value()) {
        attn_scores = attn_scores.masked_fill(mask.value() == 0, -1e9);
    }

    // 计算注意力权重并加权V
    auto attn_weights = torch::softmax(attn_scores, -1);
    return torch::matmul(attn_weights, v);
}

// 多头注意力层
class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int d_model, int n_heads)
        : d_model(d_model), n_heads(n_heads) {
        TORCH_CHECK(d_model % n_heads == 0, "d_model must be divisible by n_heads");

        d_k = d_model / n_heads;

        // 定义线性层
        w_q = register_module("w_q", torch::nn::Linear(d_model, d_model));
        w_k = register_module("w_k", torch::nn::Linear(d_model, d_model));
        w_v = register_module("w_v", torch::nn::Linear(d_model, d_model));
        w_o = register_module("w_o", torch::nn::Linear(d_model, d_model));
    }

    torch::Tensor forward(
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::optional<torch::Tensor> mask = torch::nullopt)
    {

        int batch_size = q.size(0);

        // 线性变换并拆分多头
        auto q_ = w_q(q).view({ batch_size, -1, n_heads, d_k }).transpose(1, 2);
        auto k_ = w_k(k).view({ batch_size, -1, n_heads, d_k }).transpose(1, 2);
        auto v_ = w_v(v).view({ batch_size, -1, n_heads, d_k }).transpose(1, 2);

        // 缩放点积注意力
        auto attn_output = scaled_dot_product_attention(q_, k_, v_);

        // 拼接多头并线性变换
        attn_output = attn_output.transpose(1, 2).contiguous()
            .view({ batch_size, -1, d_model });
        return w_o(attn_output);
    }

private:
    int d_model, n_heads, d_k;
    torch::nn::Linear w_q, w_k, w_v, w_o;
};

TORCH_MODULE(MultiHeadAttention);

// 前馈网络
class FeedForwardImpl : public torch::nn::Module 
{
public:
    FeedForwardImpl(int d_model, int d_ff, double dropout = 0.1)
    {
        fc1 = register_module("fc1", torch::nn::Linear(d_model, d_ff));
        fc2 = register_module("fc2", torch::nn::Linear(d_ff, d_model));
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = dropout_(x);
        x = fc2(x);
        return x;
    }

private:
    torch::nn::Linear fc1, fc2;
    torch::nn::Dropout dropout_;
};

TORCH_MODULE(FeedForward);

// Encoder层
class EncoderLayerImpl : public torch::nn::Module {
public:
    EncoderLayerImpl(int d_model, int n_heads, int d_ff, double dropout = 0.1) {
        self_attn = register_module("self_attn", MultiHeadAttention(d_model, n_heads));
        ff = register_module("ff", FeedForward(d_model, d_ff, dropout));
        norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model })));
        norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model })));
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }

    torch::Tensor forward(torch::Tensor x, torch::optional<torch::Tensor> mask = torch::nullopt) {
        // 多头自注意力 + 残差 + 层归一化
        auto attn_output = self_attn(x, x, x, mask);
        x = norm1(x + dropout_(attn_output));

        // 前馈网络 + 残差 + 层归一化
        auto ff_output = ff(x);
        x = norm2(x + dropout_(ff_output));

        return x;
    }

private:
    MultiHeadAttention self_attn;
    FeedForward ff;
    torch::nn::LayerNorm norm1, norm2;
    torch::nn::Dropout dropout_;
};

TORCH_MODULE(EncoderLayer);

// Decoder层
class DecoderLayerImpl : public torch::nn::Module {
public:
    DecoderLayerImpl(int d_model, int n_heads, int d_ff, double dropout = 0.1) {
        self_attn = register_module("self_attn", MultiHeadAttention(d_model, n_heads));
        cross_attn = register_module("cross_attn", MultiHeadAttention(d_model, n_heads));
        ff = register_module("ff", FeedForward(d_model, d_ff, dropout));
        norm1 = register_module("norm1", torch::nn::LayerNorm({ d_model }));
        norm2 = register_module("norm2", torch::nn::LayerNorm({ d_model }));
        norm3 = register_module("norm3", torch::nn::LayerNorm({ d_model }));
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }

    torch::Tensor forward(
        torch::Tensor x, torch::Tensor enc_output,
        torch::optional<torch::Tensor> tgt_mask = torch::nullopt,
        torch::optional<torch::Tensor> src_mask = torch::nullopt) {

        // 掩码自注意力（防止看到未来信息）
        auto attn_output1 = self_attn(x, x, x, tgt_mask);
        x = norm1(x + dropout_(attn_output1));

        // 编码器-解码器注意力
        auto attn_output2 = cross_attn(x, enc_output, enc_output, src_mask);
        x = norm2(x + dropout_(attn_output2));

        // 前馈网络
        auto ff_output = ff(x);
        x = norm3(x + dropout_(ff_output));

        return x;
    }

private:
    MultiHeadAttention self_attn, cross_attn;
    FeedForward ff;
    torch::nn::LayerNorm norm1, norm2, norm3;
    torch::nn::Dropout dropout_;
};

TORCH_MODULE(DecoderLayer);

// 完整Transformer
class TransformerImpl : public torch::nn::Module {
public:
    TransformerImpl(
        int src_vocab_size, int tgt_vocab_size,
        int d_model = 512, int n_heads = 8, int n_layers = 6,
        int d_ff = 2048, double dropout = 0.1, int max_len = 5000) {

        // 编码器部分
        src_embedding = register_module("src_embedding", torch::nn::Embedding(src_vocab_size, d_model));
        pos_encoding = register_module("pos_encoding", PositionalEncoding(d_model, dropout, max_len));

        // 构建Encoder层
        encoder_layers = torch::nn::ModuleList();
        for (int i = 0; i < n_layers; ++i) {
            encoder_layers->push_back(EncoderLayer(d_model, n_heads, d_ff, dropout));
        }
        register_module("encoder_layers", encoder_layers);

        // 解码器部分
        tgt_embedding = register_module("tgt_embedding", torch::nn::Embedding(tgt_vocab_size, d_model));

        // 构建Decoder层
        decoder_layers = torch::nn::ModuleList();
        for (int i = 0; i < n_layers; ++i) {
            decoder_layers->push_back(DecoderLayer(d_model, n_heads, d_ff, dropout));
        }
        register_module("decoder_layers", decoder_layers);

        // 输出层
        fc_out = register_module("fc_out", torch::nn::Linear(d_model, tgt_vocab_size));
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }

    // 生成未来掩码（防止解码器看到未来token）
    torch::Tensor generate_square_subsequent_mask(int sz) {
        auto mask = torch::triu(torch::ones({ sz, sz }), 1);
        mask = mask.masked_fill(mask == 1, 0);
        mask = mask.masked_fill(mask == 0, 1);
        return mask;
    }

    torch::Tensor forward(
        torch::Tensor src, torch::Tensor tgt,
        torch::optional<torch::Tensor> src_mask = torch::nullopt,
        torch::optional<torch::Tensor> tgt_mask = torch::nullopt,
        torch::optional<torch::Tensor> src_pad_mask = torch::nullopt,
        torch::optional<torch::Tensor> tgt_pad_mask = torch::nullopt) {

        // 编码器前向
        auto src_emb = dropout_(pos_encoding(src_embedding(src)));
        auto enc_output = src_emb;
        for (auto& layer : *encoder_layers) {
            enc_output = layer->as<EncoderLayer>()->forward(enc_output, src_pad_mask);
        }

        // 解码器前向
        auto tgt_emb = dropout_(pos_encoding(tgt_embedding(tgt)));
        auto dec_output = tgt_emb;
        for (auto& layer : *decoder_layers) {
            dec_output = layer->as<DecoderLayer>()->forward(dec_output, enc_output, tgt_mask, src_pad_mask);
        }

        // 输出层
        return fc_out(dec_output);
    }

private:
    torch::nn::Embedding src_embedding, tgt_embedding;
    PositionalEncoding pos_encoding;
    torch::nn::ModuleList encoder_layers, decoder_layers;
    torch::nn::Linear fc_out;
    torch::nn::Dropout dropout_;
};

TORCH_MODULE(Transformer);

// 测试代码
void main() {
    // 设置随机种子
    torch::manual_seed(42);

    // 超参数
    int src_vocab_size = 1000;
    int tgt_vocab_size = 2000;
    int d_model = 128;
    int n_heads = 4;
    int n_layers = 2;
    int d_ff = 512;

    // 创建Transformer实例
    Transformer transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff);
    transformer->train();  // 设置为训练模式

    // 构造测试输入（batch_size=2, src_seq_len=10, tgt_seq_len=8）
    auto src = torch::randint(0, src_vocab_size, { 2, 10 });  // [batch, src_len]
    auto tgt = torch::randint(0, tgt_vocab_size, { 2, 8 });   // [batch, tgt_len]

    // 生成未来掩码
    auto tgt_mask = transformer->generate_square_subsequent_mask(tgt.size(1));

    // 前向传播
    auto output = transformer->forward(src, tgt, torch::nullopt, tgt_mask);

    // 打印输出形状
    std::cout << "Transformer output shape: " << output.sizes() << std::endl;
    // 预期输出: [2, 8, 2000] (batch_size, tgt_seq_len, tgt_vocab_size)

    
}