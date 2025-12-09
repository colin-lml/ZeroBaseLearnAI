#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>



class SineDataset : public torch::data::Dataset<SineDataset> 
{
public:
    SineDataset(int seqlen = 3, int total_samples=1000, bool b_train = true)
    {
        seq_len = seqlen;
        bTrain = b_train;
        if (bTrain)
        {
            for (int i = 0; i < total_samples + seq_len; ++i)
            {
                data.push_back(std::sin(0.1f * i));
            }
        }
        else
        {
            for (int i = total_samples; i < total_samples + seq_len + 1; ++i)
            {
                data.push_back(std::sin(0.1f * i));
            }
        }

    }


    torch::optional<size_t> size() const 
    {
        return data.size() - seq_len;
    }


    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override 
    {
        
        std::vector<float> input_data(data.begin() + index, data.begin() + index + seq_len);
        torch::Tensor input = torch::tensor(input_data)
            .reshape({seq_len, 1 }).to(torch::kFloat32);

       
        float target_val = data[index + seq_len];
        torch::Tensor target = torch::tensor({ target_val }).to(torch::kFloat32);

        return { input, target };
    }

private:
    std::vector<float> data;  
    int seq_len = 3; 
    bool bTrain = true;
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


void RnnMain()
{
    const int seq_len = 3;       
    const int batch_size = 40;    
    const float lr = 1e-3;        
    const int epochs = 30;      
    const int total_samples = 1000;
  
    // 创建数据
    auto datasetTrain = SineDataset(seq_len, total_samples).map(torch::data::transforms::Stack<>());
    auto datasetTest = SineDataset(seq_len, total_samples,false).map(torch::data::transforms::Stack<>());
    
    // 加载数据
    auto train_data_loader = torch::data::make_data_loader(
        std::move(datasetTrain),torch::data::DataLoaderOptions().batch_size(batch_size));

    auto test_data_loader = torch::data::make_data_loader(
        std::move(datasetTest), torch::data::DataLoaderOptions().batch_size(1));


    SimpleRNN model;             
    torch::optim::Adam optimizer(model.parameters(), lr); // 
    torch::nn::MSELoss criterion; // 

 
    std::cout <<  "-------------------- 训练循环 --------------------"  << std::endl;
    model.train(); 
    for (int epoch = 0; epoch < epochs; ++epoch) 
    {
        float total_loss = 0.0f;
        int batch_count = 0;

        for (auto& batch : *train_data_loader)
        {
         
            auto y_pred = model.forward(batch.data);
        
            auto loss = criterion(y_pred, batch.target);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;
        }

    
        if (epoch % 3 == 0) 
        {
            std::cout << "Epoch " << epoch  << "/" << epochs
                      << " | Avg Loss: " << total_loss / batch_count << std::endl;
        }

        if (total_loss < 0.001)
        {
            std::cout << "total_loss < 0.001  break.Epoch " << epoch + 1 << "/" << epochs
                << " | Avg Loss: " << total_loss / batch_count << std::endl;
            break;
        }
    }


    std::cout << "-------------------- 推理测试 --------------------" << std::endl;
    model.eval(); 
    torch::NoGradGuard no_grad;
    for (auto& btest : *test_data_loader)
    {
        std::cout << std::endl << "输入序列："<<std::endl << btest.data << std::endl << std::endl;
        auto test_y_pred = model.forward(btest.data).item<float>();
        auto true_y = btest.target.item<float>();
        std::cout << "预测值：" << test_y_pred << std::endl;
        std::cout << "真实值：" << true_y << std::endl;
        std::cout << "误差：" << std::abs(test_y_pred - true_y) << std::endl;
    }
}