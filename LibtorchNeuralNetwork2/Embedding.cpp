
#include <torch/torch.h>
#include <iostream>
#include <iomanip> 


void EmbeddingMain()
{
  
    const int64_t num_embeddings = 10;     /// 字典词表大小 10个单词   
    const int64_t embedding_dim = 4;       /// 单词维度 4 
    const int64_t batch_size = 2;          
    const int64_t seq_len = 3;             
    const int64_t epochs = 50;             
    const float lr = 0.1;                  

    auto model = torch::nn::Embedding(torch::nn::EmbeddingOptions(num_embeddings, embedding_dim));
 
    torch::nn::MSELoss loss_fn;
    
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(lr));
    /// 1, 3, 5， 2, 4, 6   字典词表 索引位置   
    torch::Tensor input_indices = torch::tensor({{1, 3, 5},{2, 4, 6}}, torch::kLong);
    
    torch::Tensor target_embeddings = torch::tensor({
        {{0.1, 0.2, 0.3, 0.4}, {0.3, 0.4, 0.5, 0.6}, {0.5, 0.6, 0.7, 0.8}}, ///  目标 词向量
        {{0.2, 0.3, 0.4, 0.5}, {0.4, 0.5, 0.6, 0.7}, {0.6, 0.7, 0.8, 0.9}}  ///  目标 词向量
        }, torch::kFloat);

    
    std::cout << "=== 训练前嵌入矩阵 ===" << std::endl;
    auto init_embedding = model->weight.data();
    std::cout << init_embedding << std::endl;


    std::cout << "\n=== 开始训练 ===" << std::endl;
    model->train();  

    for (int64_t epoch = 0; epoch < epochs; ++epoch) 
    {
        
        optimizer.zero_grad();
        torch::Tensor output = model->forward(input_indices);
        torch::Tensor loss = loss_fn(output, target_embeddings);
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 5 == 0)
        {
            std::cout << "Epoch: " << std::setw(3) << (epoch + 1)
                << " | Loss: " << std::fixed << std::setprecision(6) << loss.item<float>() << std::endl;
        }
    }

   
    std::cout << "\n=== 训练后嵌入矩阵 ===" << std::endl;
    auto final_embedding = model->weight.data();
    std::cout << final_embedding << std::endl;

    std::cout << "\n=== 训练后推理示例 ===" << std::endl;
    torch::Tensor test_input = torch::tensor({ 1, 3 }, torch::kLong);  
    torch::Tensor test_output = model->forward(test_input);
    std::cout << "测试输入索引: [1, 3]" << std::endl;
    std::cout << "训练后嵌入输出:\n" << test_output << std::endl;

    
}