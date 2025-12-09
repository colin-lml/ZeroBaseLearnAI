#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>



class SineDataset : public torch::data::Dataset<SineDataset> {
public:
    SineDataset(int seq_len = 3, int total_samples=1000)
        : seq_len_(seq_len) 
    {
  
        for (int i = 0; i < total_samples + seq_len; ++i)
        {
            data_.push_back(std::sin(0.3f * i)); 
        }
    }


    torch::optional<size_t> size() const 
    {
        return data_.size() / seq_len_;
    }


    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override 
    {
        
        std::vector<float> input_data(data_.begin() + index, data_.begin() + index + seq_len_);
        torch::Tensor input = torch::tensor(input_data)
            .reshape({1, seq_len_, 1 })  // [n,seq_len, input_size=1]
            .to(torch::kFloat32);

       
        float target_val = data_[index + seq_len_];
        torch::Tensor target = torch::tensor({ target_val }).to(torch::kFloat32);

        return { input, target };
    }

private:
    std::vector<float> data_;  
    int seq_len_;              
};



class SimpleRNN : public torch::nn::Module
{
public:
    SimpleRNN(int input_size = 1, int hidden_size = 16, int output_size = 1)
        : hidden_size_(hidden_size) 
    {
      
        torch::nn::RNNOptions rnn_options(input_size, hidden_size);
        rnn_options.batch_first(true); 

        rnn_ = register_module("rnn", torch::nn::RNN(rnn_options));
        
        fc_ = register_module("fc", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(const torch::Tensor& x) 
    {
        auto [outputs, hidden] = rnn_->forward(x);
        torch::Tensor last_hidden = hidden.squeeze(0).squeeze(0); 
        return fc_->forward(last_hidden);
    }

private:
    torch::nn::RNN rnn_{ nullptr };          
    torch::nn::Linear fc_{nullptr};        
    int hidden_size_; 
};

// ===================== 3. 训练 + 推理主逻辑 =====================
void RnnMain()
{
 
    
    const int seq_len = 3;       
    const int batch_size = 32;    
    const float lr = 1e-3;        
    const int epochs = 30;      

    // -------------------- 初始化数据集和数据加载器 --------------------
    auto dataset = SineDataset(seq_len, 1000);
    
    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),torch::data::DataLoaderOptions().batch_size(batch_size));

    SimpleRNN model;             
    torch::optim::Adam optimizer(model.parameters(), lr); // 
    torch::nn::MSELoss criterion; // 均方误差损失（回归任务）

    // -------------------- 训练循环 --------------------
    model.train(); 
    for (int epoch = 0; epoch < epochs; ++epoch) 
    {
        float total_loss = 0.0f;
        int batch_count = 0;

        for (auto& batch : *data_loader) 
        {
            auto x = batch.data()->data;
            auto y = batch.data()->target;
   
            auto y_pred = model.forward(x);
        
            auto loss = criterion(y_pred, y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;
        }

    
        if ((epoch + 1) % 5 == 0) 
        {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << " | Avg Loss: " << total_loss / batch_count << std::endl;
        }
    }

    // -------------------- 推理测试 --------------------
    model.eval(); 
    std::cout << "\n=== 推理测试 ===" << std::endl;

    
    std::vector<float> test_seq = {
        std::sin(0.3f * 1000),
        std::sin(0.3f * 1001),
        std::sin(0.3f * 1002)
    };
    torch::Tensor test_x = torch::tensor(test_seq)
        .reshape({1, seq_len, 1}) 
        .to(torch::kFloat32);

  
    torch::NoGradGuard no_grad;
    auto test_y_pred = model.forward(test_x);

   
    float true_y = test_seq[2];
    std::cout << "输入序列：[" << test_seq[0] << ", " << test_seq[1] << ", " << test_seq[2] << "]" << std::endl;
    std::cout << "预测值：" << test_y_pred.item<float>() << std::endl;
    std::cout << "真实值：" << true_y << std::endl;
    std::cout << "误差：" << std::abs(test_y_pred.item<float>() - true_y) << std::endl;

}