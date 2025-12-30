#include <torch/torch.h>
#include <iostream>

// 手动实现 Scaled Dot-Product Attention（QKV 核心）
class ScaledDotProductAttentionImpl : public torch::nn::Module {
public:
    ScaledDotProductAttentionImpl() = default;

    // 前向计算：核心 QKV 逻辑
    // q: [seq_len_q, batch_size, d_k]
    // k: [seq_len_k, batch_size, d_k]
    // v: [seq_len_v, batch_size, d_v] (seq_len_k == seq_len_v)
    // mask: [seq_len_q, seq_len_k] 或 nullptr（可选）
    torch::Tensor forward(
        const torch::Tensor& q,
        const torch::Tensor& k,
        const torch::Tensor& v,
        const torch::optional<torch::Tensor>& mask = torch::nullopt
    ) 
    {
        int64_t d_k = q.size(-1);

        // 1. 计算 Q * K^T (转置最后两个维度)
        // [seq_len_q, batch_size, d_k] * [batch_size, d_k, seq_len_k] = [seq_len_q, batch_size, seq_len_k]
       
        auto scores = torch::matmul(q, k.transpose(0, 2));

        // 2. 缩放（除以 sqrt(d_k)）
        scores = scores / std::sqrt(d_k);

        // 3. 掩码处理（将需要屏蔽的位置设为极小值，Softmax后接近0）
        if (mask.has_value()) 
        {
            scores = scores.masked_fill(mask.value() == 0, -1e9);
        }

        // 4. Softmax 计算注意力权重
        auto attn_weights = torch::softmax(scores, /*dim=*/-1);

        // 5. 加权求和 V
        // [seq_len_q, batch_size, seq_len_k] * [seq_len_k, batch_size, d_v] = [seq_len_q, batch_size, d_v]
        auto output = torch::matmul(attn_weights, v);

        return output;
    }
};

TORCH_MODULE(ScaledDotProductAttention);

// 实现多头注意力（Multi-Head Attention）
class MultiHeadAttention : public torch::nn::Module {
public:
    MultiHeadAttention(
        int64_t d_model,    // 模型总维度
        int64_t nhead,      // 注意力头数
        double dropout = 0.1
    ) : d_model(d_model), nhead(nhead), d_k(d_model / nhead) 
    {
        // 验证维度合法性
        TORCH_CHECK(d_model % nhead == 0, "d_model 必须能被 nhead 整除");

        // 定义 QKV 投影层
        w_q = register_module("w_q", torch::nn::Linear(d_model, d_model));
        w_k = register_module("w_k", torch::nn::Linear(d_model, d_model));
        w_v = register_module("w_v", torch::nn::Linear(d_model, d_model));
        w_o = register_module("w_o", torch::nn::Linear(d_model, d_model));

        // 注意力层
        attention = register_module("attention", ScaledDotProductAttention());

        // Dropout 层
        dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));

        // 初始化权重
        torch::nn::init::xavier_uniform_(w_q->weight);
        torch::nn::init::xavier_uniform_(w_k->weight);
        torch::nn::init::xavier_uniform_(w_v->weight);
        torch::nn::init::xavier_uniform_(w_o->weight);
        torch::nn::init::zeros_(w_q->bias);
        torch::nn::init::zeros_(w_k->bias);
        torch::nn::init::zeros_(w_v->bias);
        torch::nn::init::zeros_(w_o->bias);
    }

    // 将 QKV 拆分为多个头
    // x: [seq_len, batch_size, d_model] -> [seq_len, batch_size * nhead, d_k]
    torch::Tensor split_heads(const torch::Tensor& x)
    {
        auto new_shape = torch::IntArrayRef({ x.size(0), x.size(1) * nhead, d_k });
        return x.view({ x.size(0), x.size(1), nhead, d_k })
            .permute({ 0, 2, 1, 3 })
            .reshape(new_shape);
    }

    // 将多个头的结果合并
    // x: [seq_len, batch_size * nhead, d_k] -> [seq_len, batch_size, d_model]
    torch::Tensor combine_heads(const torch::Tensor& x) {
        auto new_shape = torch::IntArrayRef({ x.size(0), -1, nhead * d_k });
        return x.view({ x.size(0), nhead, -1, d_k })
            .permute({ 0, 2, 1, 3 })
            .reshape(new_shape);
    }

    // 前向计算
    torch::Tensor forward(
        const torch::Tensor& query,  // [seq_len_q, batch_size, d_model]
        const torch::Tensor& key,    // [seq_len_k, batch_size, d_model]
        const torch::Tensor& value,  // [seq_len_v, batch_size, d_model]
        const torch::optional<torch::Tensor>& mask = torch::nullopt
    ) 
    {
        int64_t batch_size = query.size(1);

        // 1. QKV 线性投影
        auto q = w_q(query);  // [seq_len_q, batch_size, d_model]
        auto k = w_k(key);    // [seq_len_k, batch_size, d_model]
        auto v = w_v(value);  // [seq_len_v, batch_size, d_model]

        // 2. 拆分为多个头
        q = split_heads(q);   // [seq_len_q, batch_size*nhead, d_k]
        k = split_heads(k);   // [seq_len_k, batch_size*nhead, d_k]
        v = split_heads(v);   // [seq_len_v, batch_size*nhead, d_v]

        // 3. 计算自注意力
        auto attn_output = attention->forward(q, k, v, mask);  // [seq_len_q, batch_size*nhead, d_v]
    
        // 4. 合并多头结果
        attn_output = combine_heads(attn_output);  // [seq_len_q, batch_size, d_model]

        // 5. 输出投影 + Dropout
        auto output = dropout_layer(w_o(attn_output));

        return output;
    }

private:
    int64_t d_model;          // 模型总维度
    int64_t nhead;            // 注意力头数
    int64_t d_k;              // 每个头的维度

    torch::nn::Linear w_q{ nullptr };  // Query 投影层
    torch::nn::Linear w_k{ nullptr };  // Key 投影层
    torch::nn::Linear w_v{ nullptr };  // Value 投影层
    torch::nn::Linear w_o{ nullptr };  // 输出投影层

    torch::nn::Dropout dropout_layer{ nullptr };
    ScaledDotProductAttention attention{ nullptr };
};

// 测试代码
void TransformerMain() 
{

    torch::Tensor t2d = torch::tensor({
     {1, 2, 3},
     {4, 5, 6}
        });
    std::cout << "=== 2维张量原始数据 ===" << std::endl;
    std::cout << "形状: " << t2d.sizes() << "\n" << t2d << std::endl;

    // transpose(0, 1)：交换行（维度0）和列（维度1），等价于 t2d.t()
    torch::Tensor t2d_trans = t2d.transpose(0, 1);
    std::cout << "\ntranspose(0, 1) 后（等价于 t()）:" << std::endl;
    std::cout << "形状: " << t2d_trans.sizes() << "\n" << t2d_trans << std::endl;


     
    torch::nn::TransformerOptions opt;
    opt.dim_feedforward();  // dim_feedforward 前馈网络层  2048
    opt.activation(); // 编/解码器中间层的激活功能    torch::kReLU
    
    torch::nn::Transformer transformer(opt);  // 创建 Transformer
    auto  src_emb = torch::Tensor();
    auto  tgt_emb = torch::Tensor();
    auto  tgt_mask = torch::Tensor();
    auto transout = transformer->forward(
        src_emb,          // 编码器输入
        tgt_emb,          // 解码器输入
        tgt_mask,         // 解码器自注意力掩码
        torch::Tensor(),  // 编码器自注意力掩码（无掩码）
        torch::Tensor(),  // 编码器-解码器注意力掩码（无掩码）
        torch::Tensor(),  // 源序列padding掩码（无padding）
        torch::Tensor(),  // 目标序列padding掩码（无padding）
        torch::Tensor()   // 编码器-解码器padding掩码（无padding）
    );




    // 配置参数
    const int64_t seq_len = 10;     // 序列长度
    const int64_t batch_size = 2;   // 批次大小
    const int64_t d_model = 128;    // 模型维度
    const int64_t nhead = 8;        // 注意力头数

    // 设置设备（优先GPU）
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "使用设备: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // 1. 创建多头注意力层
    MultiHeadAttention mha(d_model, nhead);
    mha.to(device);

    // 2. 生成测试数据 [seq_len, batch_size, d_model]
    torch::Tensor x = torch::randn({ seq_len, batch_size, d_model }, torch::kFloat).to(device);

    // 3. 生成掩码（示例：屏蔽最后3个位置）
    torch::Tensor mask = torch::ones({ seq_len, seq_len }, torch::kBool).to(device);
    mask.slice(1, seq_len - 3, seq_len) = 0;  // 最后3列设为0（屏蔽）

    // 4. 前向计算（自注意力：Q=K=V=x）
    mha.eval();
    torch::NoGradGuard no_grad;
    std::cout << "\n输入形状: " << x.sizes() << std::endl;
    std::cout << "\n掩码形状: " << mask.sizes() << std::endl;
    auto output = mha.forward(x, x, x, mask);

    // 5. 打印结果信息
    std::cout << "\n输入形状: " << x.sizes() << std::endl;
    std::cout << "输出形状: " << output.sizes() << std::endl;
    std::cout << "\n输出前5个值:\n" << output[0][0].slice(0, 0, 5) << std::endl;

   
}