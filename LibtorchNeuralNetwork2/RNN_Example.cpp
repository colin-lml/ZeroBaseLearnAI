#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

#if 0
void RnnMain() 
{

    torch::Device device = torch::kCPU;

    torch::nn::RNNOptions rnn_options(
        10,    // input_size: 输入特征维度（每个时间步的输入向量长度）
        20     // hidden_size: 隐藏层维度（每个时间步的输出向量长度）
    );


    // 初始化 RNN 模型并移到设备
    torch::nn::RNN rnn_model(rnn_options);
    rnn_model->to(device);

    // ======================================
    // 3. 构造输入数据
    // ======================================
    const int seq_len = 5;    // 序列长度（时间步数量）
    const int batch_size = 3; // 批次大小
    const int input_size = 10; // 必须与 RNN 输入维度一致

    torch::Tensor input = torch::randn({ seq_len, batch_size, input_size }, device);

  
    torch::Tensor h0 = torch::randn({ 1, batch_size, 20 }, device);

 
    // ======================================
    rnn_model->train(); // 设置训练模式（默认也是训练模式）
    auto output = rnn_model->forward(input, h0);

    // ======================================
    // 5. 输出结果解析
    // ======================================
    torch::Tensor out = std::get<0>(output); // 所有时间步的输出：(seq_len, batch_size, hidden_size)
    torch::Tensor hn = std::get<1>(output); // 最后一个时间步的隐藏状态：(num_layers, batch_size, hidden_size)

    
    std::cout << "所有时间步输出形状: " << out.sizes() << std::endl; // 输出: [5, 3, 20]
    std::cout << "最后隐藏状态形状: " << hn.sizes() << std::endl;  // 输出: [1, 3, 20]

 
    torch::Tensor last_time_step_out = out.index({ seq_len - 1, torch::indexing::Slice() });
    std::cout << "\n最后时间步输出与最后隐藏状态是否近似相等: "
        << torch::allclose(last_time_step_out, hn.squeeze(0), 1e-5) << std::endl;

   
}
#endif


#if 0

// 1. 生成正弦波序列数据（输入序列+目标值）
// input_seq_len: 输入序列长度（如3个连续时间步）
// total_samples: 总样本数
// 返回: (inputs, targets) -> inputs: [N, input_seq_len, 1], targets: [N, 1]
std::pair<torch::Tensor, torch::Tensor> generate_sine_data(int input_seq_len = 3, int total_samples = 1000) 
{
    std::vector<float> x_data, y_data;
    // 生成0到10π的正弦波数据
    for (int i = 0; i < total_samples + input_seq_len; ++i) 
    {
        float x = 0.1f * i;  // 步长0.1，覆盖0~100π（足够长的序列）
        x_data.push_back(std::sin(x));
    }
    // 构造输入序列和目标值（目标=输入序列的下一个时间步）
    std::vector<torch::Tensor> inputs, targets;
    for (int i = 0; i < total_samples; ++i) 
    {
        // 输入序列：[i, i+1, i+2] -> 形状[input_seq_len, 1]
        auto input_seq = torch::tensor(std::vector<float>(x_data.begin() + i, x_data.begin() + i + input_seq_len)).reshape({ input_seq_len, 1 }).to(torch::kFloat32);
        // 目标值：i+3时刻的正弦值 -> 形状[1]
        auto target = torch::tensor(x_data[i + input_seq_len]).reshape({ 1 }).to(torch::kFloat32);
        inputs.push_back(input_seq);
        targets.push_back(target);
    }
    // 拼接为批量张量：inputs[N, input_seq_len, 1]，targets[N, 1]
    return 
    {
        torch::stack(inputs),
        torch::stack(targets)
    };
}

// 2. 定义简单RNN模型
class SimpleRNN : public torch::nn::Module {
public:
    // 构造函数：input_size=1（每个时间步输入维度），hidden_size=32（隐藏层维度），output_size=1（输出维度）
    SimpleRNN(int input_size = 1, int hidden_size = 32, int output_size = 1)
        : hidden_size_(hidden_size) {
        // RNN层：input_size -> hidden_size，batch_first=true（输入格式[N, seq_len, input_size]）
        rnn_ = register_module("rnn", torch::nn::RNN(
            torch::nn::RNNOptions(input_size, hidden_size).batch_first(true)
        ));
        // 全连接层：hidden_size -> output_size（将RNN输出映射到预测值）
        fc_ = register_module("fc", torch::nn::Linear(hidden_size, output_size));
    }

    // 前向传播：x -> [N, seq_len, input_size]
    torch::Tensor forward(const torch::Tensor& x) {
        // RNN输出：outputs[N, seq_len, hidden_size]，hidden[N, hidden_size]（最后一个时间步的隐藏状态）
        auto [outputs, hidden] = rnn_->forward(x);

        // 取最后一个时间步的输出用于预测（因为目标是下一个时间步，与最后一个状态最相关）
        torch::Tensor last_hidden = outputs.index({ "...", -1, "..." });  // [N, hidden_size]

        // 全连接层映射到输出维度
        torch::Tensor out = fc_->forward(last_hidden);  // [N, output_size]
        return out;
    }

private:
    torch::nn::RNN rnn_;
    torch::nn::Linear fc_;
    int hidden_size_;
};

class CustomRnnDataset : public torch::data::datasets::Dataset<CustomRnnDataset>
{
    std::vector<float>  data;

    using Example = torch::data::Example<>;
    Example get(size_t index)
    {

    }

    torch::optional<size_t> size() const 
    {
        return data.size();
    }
};



void RnnMain() 
{
    // ---------------------- 配置参数 ----------------------
    const int input_seq_len = 3;    // 输入序列长度（用前3个点预测下1个）
    const int input_size = 1;       // 每个时间步输入维度（正弦波是单变量）
    const int hidden_size = 32;     // RNN隐藏层维度
    const int output_size = 1;      // 输出维度（预测1个值）
    const int batch_size = 32;      // 批量大小
    const float lr = 1e-3;          // 学习率
    const int epochs = 50;          // 训练轮数

    // ---------------------- 生成数据 ----------------------
    auto [train_inputs, train_targets] = generate_sine_data(input_seq_len, 1000);
    std::cout << "训练数据形状：" << std::endl;
    std::cout << "inputs: " << train_inputs.sizes() << " (N, seq_len, input_size)" << std::endl;
    std::cout << "targets: " << train_targets.sizes() << " (N, output_size)" << std::endl;

    // 构造数据集和数据加载器（批量迭代）
    auto dataset = torch::data::TensorDataset(train_inputs, train_targets);
    auto data_loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size)  // 打乱数据
    );

    // ---------------------- 初始化模型、优化器、损失函数 ----------------------
    SimpleRNN model(input_size, hidden_size, output_size);
    torch::optim::Adam optimizer(model.parameters(), lr);  // Adam优化器
    torch::nn::MSELoss criterion;  // 均方误差损失（回归任务）

    // 设置训练模式
    model.train();

    // ---------------------- 训练过程 ----------------------
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        float total_loss = 0.0f;
        int batch_count = 0;

        // 迭代每个批量
        for (auto& batch : *data_loader) {
            auto [x, y] = batch;  // x: [batch_size, seq_len, 1], y: [batch_size, 1]

            // 前向传播
            auto y_pred = model.forward(x);

            // 计算损失
            auto loss = criterion(y_pred, y);

            // 反向传播 + 梯度更新
            optimizer.zero_grad();  // 清零梯度
            loss.backward();        // 反向传播
            optimizer.step();       // 更新参数

            // 累计损失
            total_loss += loss;
            batch_count++;
        }

        // 打印每轮平均损失
        float avg_loss = total_loss / batch_count;
        if ((epoch + 1) % 5 == 0) {  // 每5轮打印一次
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                << ", Avg Loss: " << avg_loss << std::endl;
        }
    }

    // ---------------------- 推理测试 ----------------------
    model.eval();  // 设置评估模式（禁用Dropout等）
    std::cout << "\n=== 推理测试 ===" << std::endl;

    // 手动构造一个测试序列（前3个点），预测第4个点
    std::vector<float> test_seq_data = {
        std::sin(0.1f * 1000),   // 第1000个时间步
        std::sin(0.1f * 1001),   // 第1001个时间步
        std::sin(0.1f * 1002)    // 第1002个时间步
    };
    auto test_x = torch::tensor(test_seq_data)
        .reshape({ 1, input_seq_len, input_size })  // [1, 3, 1]（批量大小=1）
        .to(torch::kFloat32);

    // 预测
    auto test_y_pred = model.forward(test_x);
    float true_y = std::sin(0.1f * 1003);  // 第1003个时间步的真实值

    // 打印结果
    std::cout << "输入序列：[" << test_seq_data[0] << ", " << test_seq_data[1] << ", " << test_seq_data[2] << "]" << std::endl;
    std::cout << "预测值：" << test_y_pred.item<float>() << std::endl;
    std::cout << "真实值：" << true_y << std::endl;
    std::cout << "误差：" << std::abs(test_y_pred.item<float>() - true_y) << std::endl;
  
}

#endif


// ===================== 1. 定义Embedding+RNN模型 =====================
class EmbeddingRNNClassifier : public torch::nn::Module {
public:
    // 构造函数：初始化嵌入层、RNN层、分类头
    // vocab_size: 词汇表大小（离散单词ID总数）
    // embed_dim: 词向量维度
    // hidden_size: RNN隐藏层维度
    // num_classes: 分类类别数
    EmbeddingRNNClassifier(
        int64_t vocab_size,
        int64_t embed_dim,
        int64_t hidden_size,
        int64_t num_classes
    ) :
        embedding_(torch::nn::EmbeddingOptions(vocab_size, embed_dim).padding_idx(0)), // 0为填充位
        rnn_(torch::nn::RNNOptions(embed_dim, hidden_size)
            .num_layers(1)          // 单层RNN
            .batch_first(true)      // 输入格式：(batch_size, seq_len, embed_dim)（更直观）
            .bidirectional(false)), // 单向RNN
        fc_(hidden_size, num_classes) { // 分类全连接层

        // 注册子模块（必须！否则参数无法被优化器捕获）
        register_module("embedding", embedding_);
        register_module("rnn", rnn_);
        register_module("fc", fc_);
    }

    // 前向传播
    torch::Tensor forward(torch::Tensor x) {
        // x: (batch_size, seq_len) → 输入为单词ID序列

        // Step 1: Embedding层 → 词向量序列
        // output: (batch_size, seq_len, embed_dim)
        torch::Tensor embed = embedding_->forward(x);

        // Step 2: RNN层提取时序特征
        // 初始化隐藏状态h0: (num_layers * num_directions, batch_size, hidden_size)
        auto h0 = torch::zeros({ 1, x.size(0), rnn_->options.hidden_size() },
            torch::device(embed.device()).dtype(embed.dtype()));
        // RNN输出：(output, hn)
        // output: (batch_size, seq_len, hidden_size) → 所有时间步输出
        // hn: (1, batch_size, hidden_size) → 最后时刻隐藏状态
        auto rnn_out = rnn_->forward(embed, h0);
        torch::Tensor hn = std::get<1>(rnn_out); // 提取最后时刻隐藏状态

        // Step 3: 分类头（去掉num_layers维度）
        // hn.squeeze(0): (batch_size, hidden_size)
        torch::Tensor logits = fc_(hn.squeeze(0)); // 输出：(batch_size, num_classes)

        return logits;
    }

private:
    torch::nn::Embedding embedding_; // 嵌入层
    torch::nn::RNN rnn_;             // RNN层
    torch::nn::Linear fc_;           // 分类全连接层
};

// ===================== 2. 生成模拟文本序列数据 =====================
// 生成：(batch_size, seq_len)的单词ID序列 + (batch_size,)的标签
std::pair<torch::Tensor, torch::Tensor> generate_text_data(
    int64_t batch_size,
    int64_t seq_len,
    int64_t vocab_size,
    int64_t num_classes
) {
    // 单词ID序列：值范围0~vocab_size-1（0为填充位）
    torch::Tensor input_ids = torch::randint(0, vocab_size, { batch_size, seq_len }, torch::kLong);
    // 分类标签：0/1（二分类）
    torch::Tensor labels = torch::randint(0, num_classes, { batch_size }, torch::kLong);
    return { input_ids, labels };
}

// ===================== 3. 主函数（训练+预测） =====================
void RnnMain() {
    // -------------------- 超参数设置 --------------------
    const int64_t vocab_size = 1000;    // 词汇表大小（单词ID：0~999）
    const int64_t embed_dim = 64;       // 词向量维度
    const int64_t hidden_size = 128;    // RNN隐藏层维度
    const int64_t num_classes = 2;      // 二分类
    const int64_t seq_len = 15;         // 序列长度（每个文本15个单词）
    const int64_t batch_size = 16;      // 批次大小
    const int64_t epochs = 30;          // 训练轮数
    const float lr = 0.001f;            // 学习率

    // -------------------- 初始化模型/优化器/损失函数 --------------------
    EmbeddingRNNClassifier model(vocab_size, embed_dim, hidden_size, num_classes);
    // 优化器：Adam（适配嵌入层+RNN的参数更新）
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lr));
    // 损失函数：交叉熵（适配分类任务）
    torch::nn::CrossEntropyLoss criterion;

    // -------------------- 训练循环 --------------------
    model.train(); // 训练模式
    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        // 生成一批训练数据
        auto [input_ids, labels] = generate_text_data(batch_size, seq_len, vocab_size, num_classes);

        // 前向传播
        optimizer.zero_grad(); // 梯度清零
        torch::Tensor logits = model.forward(input_ids);

        // 计算损失
        torch::Tensor loss = criterion(logits, labels);

        // 反向传播 + 更新参数
        loss.backward();
        optimizer.step();

        // 打印训练信息（每5轮）
        if ((epoch + 1) % 5 == 0) {
            // 计算准确率
            auto preds = logits.argmax(1); // 预测类别：(batch_size,)
            float acc = preds.eq(labels).sum().item<float>() / batch_size;

            std::cout << "Epoch: " << epoch + 1
                << " | Loss: " << loss.item<float>()
                << " | Acc: " << acc << std::endl;
        }
    }

    // -------------------- 预测示例 --------------------
    model.eval(); // 评估模式
    torch::NoGradGuard no_grad; // 禁用梯度计算（提升推理效率）

    // 生成单个测试样本（batch_size=1）
    auto [test_ids, test_label] = generate_text_data(1, seq_len, vocab_size, num_classes);
    torch::Tensor test_logits = model.forward(test_ids);
    auto pred_label = test_logits.argmax(1).item<int64_t>();
    auto true_label = test_label.item<int64_t>();

    std::cout << "\n=== 预测结果 ===" << std::endl;
    std::cout << "输入单词ID序列:\n" << test_ids.squeeze(0) << std::endl;
    std::cout << "真实标签: " << true_label << std::endl;
    std::cout << "预测标签: " << pred_label << std::endl;


}