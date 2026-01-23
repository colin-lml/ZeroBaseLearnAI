#include <torch/torch.h>
#include <iostream>

// ===================== 1. 工具函数：封装卷积层（减少冗余） =====================
torch::nn::Conv2d conv3x3(int in_c, int out_c, int stride = 1) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_c, out_c, 3)
        .stride(stride)
        .padding(1)  // 3x3卷积padding=1，保证尺寸不变
        .bias(false));  // 后接BN，关闭偏置
}

torch::nn::Conv2d conv1x1(int in_c, int out_c, int stride = 1) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_c, out_c, 1)
        .stride(stride)
        .bias(false));
}

// ===================== 2. 基础残差块（BasicBlock）：ResNet核心单元 =====================
struct BasicBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1, conv2;
    torch::nn::BatchNorm2d bn1, bn2;
    torch::nn::Sequential shortcut;  // 捷径连接（维度不匹配时用）

    // 构造函数：in_c=输入通道，out_c=输出通道，stride=步长（调整尺寸）
    BasicBlockImpl(int in_c, int out_c, int stride = 1) {
        // 主路径：Conv → BN → ReLU → Conv → BN
        conv1 = register_module("conv1", conv3x3(in_c, out_c, stride));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_c));
        conv2 = register_module("conv2", conv3x3(out_c, out_c));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_c));

        // 捷径连接：维度不匹配时（通道/尺寸不同），用1x1卷积调整
        if (stride != 1 || in_c != out_c) {
            shortcut = register_module("shortcut", torch::nn::Sequential(
                conv1x1(in_c, out_c, stride),
                torch::nn::BatchNorm2d(out_c)
            ));
        }
    }

    // 前向传播：核心逻辑 F(x) + x
    torch::Tensor forward(torch::Tensor x) {
        // 1. 计算残差F(x)
        auto residual = torch::relu(bn1(conv1(x)));
        residual = bn2(conv2(residual));
        // 2. 残差 + 捷径连接（维度匹配直接加，否则加调整后的x）
        residual += shortcut.empty() ? x : shortcut->forward(x);
        
        // 3. 最终激活
        return torch::relu(residual);
    }
};
// 封装为ModuleHolder，方便后续使用
TORCH_MODULE(BasicBlock);

// ===================== 3. 简化版ResNet（适配MNIST） =====================
struct SimpleResNetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1;       // 初始卷积层
    torch::nn::BatchNorm2d bn1;    // 初始BN层
    torch::nn::Sequential layer1;  // 残差块组1（2个BasicBlock）
    torch::nn::Linear fc;          // 全连接层（输出10类）

    SimpleResNetImpl() {
        // 初始层：7x7卷积→BN→ReLU→池化（适配28x28输入）
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 64, 7)  // 1通道输入（MNIST），64通道输出
            .stride(2)
            .padding(3)
            .bias(false)
        ));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        // 残差块组1：2个BasicBlock（64通道，步长1）
        layer1 = register_module("layer1", torch::nn::Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        ));

        // 全连接层：64*7*7（池化后尺寸）→ 10类
        fc = register_module("fc", torch::nn::Linear(64 * 7 * 7, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        // 1. 初始层
        x = torch::relu(bn1(conv1(x)));
        x = torch::max_pool2d(x, 3, 2, 1);  // 28x28→14x14→7x7（池化后）

        // 2. 残差块组
        x = layer1->forward(x);

        // 3. 全局平均池化 + 展平 + 全连接
        x = torch::avg_pool2d(x, { x.size(2), x.size(3) });  // 7x7→1x1
        x = x.view({ x.size(0), -1 });  // 展平：(batch, 64)
        x = fc(x);

        return x;
    }
};
TORCH_MODULE(SimpleResNet);

// ===================== 4. 测试代码：验证网络正确性 =====================
int main() {
    // 1. 设备配置（优先CUDA）
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "使用设备：" << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // 2. 初始化简化版ResNet
    SimpleResNet resnet;
    resnet->to(device);  // 移到指定设备

    // 3. 构造MNIST测试输入：batch=4，1通道，28x28
    torch::Tensor x = torch::randn({ 4, 1, 28, 28 }).to(device);
    std::cout << "输入形状：" << x.sizes() << std::endl;

    // 4. 前向传播
    torch::Tensor out = resnet->forward(x);

    // 5. 输出结果
    std::cout << "输出形状：" << out.sizes() << std::endl;  // 预期：[4, 10]
    std::cout << "输出示例（第一个样本的10类概率）：\n"
        << out[0].detach().cpu() << std::endl;

    return 0;
}