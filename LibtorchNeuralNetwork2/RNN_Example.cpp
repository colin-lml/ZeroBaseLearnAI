#include <torch/torch.h>
#include <iostream>


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

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// 1. 生成正弦波序列数据（输入序列+目标值）
// input_seq_len: 输入序列长度（如3个连续时间步）
// total_samples: 总样本数
// 返回: (inputs, targets) -> inputs: [N, input_seq_len, 1], targets: [N, 1]
std::pair<torch::Tensor, torch::Tensor> generate_sine_data(int input_seq_len = 3, int total_samples = 1000) {
    std::vector<float> x_data, y_data;
    // 生成0到10π的正弦波数据
    for (int i = 0; i < total_samples + input_seq_len; ++i) {
        float x = 0.1f * i;  // 步长0.1，覆盖0~100π（足够长的序列）
        x_data.push_back(std::sin(x));
    }
    // 构造输入序列和目标值（目标=输入序列的下一个时间步）
    std::vector<torch::Tensor> inputs, targets;
    for (int i = 0; i < total_samples; ++i) {
        // 输入序列：[i, i+1, i+2] -> 形状[input_seq_len, 1]
        auto input_seq = torch::tensor(std::vector<float>(
            x_data.begin() + i, x_data.begin() + i + input_seq_len
        )).reshape({ input_seq_len, 1 }).to(torch::kFloat32);
        // 目标值：i+3时刻的正弦值 -> 形状[1]
        auto target = torch::tensor(x_data[i + input_seq_len]).reshape({ 1 }).to(torch::kFloat32);
        inputs.push_back(input_seq);
        targets.push_back(target);
    }
    // 拼接为批量张量：inputs[N, input_seq_len, 1]，targets[N, 1]
    return {
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

int main() {
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
        torch::data::DataLoaderOptions().batch_size(batch_size).shuffle(true)  // 打乱数据
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
            total_loss += loss.item<float>();
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

    return 0;
}