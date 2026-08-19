#pragma once

class PolicyNetContImpl : public torch::nn::Module
{
public:

    PolicyNetContImpl() = default;
    PolicyNetContImpl(int64_t input,int64_t output, double actionBound, int64_t hidden = 128)
    {
        m_fc1 = register_module("fc1",torch::nn::Linear(input, hidden));
        m_fc2 = register_module("fc2",torch::nn::Linear(hidden, output));
        m_dbActionBound = actionBound;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
       
        return m_dbActionBound * torch::tanh(m_fc2->forward(x));
    }

private:
    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_fc2{ nullptr };
    double m_dbActionBound = 2.0;
};

TORCH_MODULE(PolicyNetCont);


class QValueNetContImpl : public torch::nn::Module
{
public:
    QValueNetContImpl() = default;

    QValueNetContImpl(int64_t input,int64_t output,int64_t hidden = 128)
    {
        m_fc1 = register_module("fc1", torch::nn::Linear(input + output, hidden));

        m_fc2 = register_module("fc2",torch::nn::Linear(hidden, hidden));

        m_output = register_module("output",torch::nn::Linear(hidden, 1));
    }

    torch::Tensor forward( const torch::Tensor& state,const torch::Tensor& action)
    {
        auto x = torch::cat({ state, action }, 1);

        x = torch::relu(m_fc1->forward(x));
        x = torch::relu(m_fc2->forward(x));

        return m_output->forward(x);
    }

private:
    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_fc2{ nullptr };
    torch::nn::Linear m_output{ nullptr };
};

TORCH_MODULE(QValueNetCont);


class DDPG : public BaseAdvanced
{
public:
    DDPG(): BaseAdvanced(false){}
private:
    double TakeAction(VectorDouble& s0, bool bPredict = false) override;

    void GenerateTrainData(int maxCount) override;

    void TrainGenerateItem1(const QwItem& item) override;

    void TrainGenerateItem2(const QwList& vList) override;


    void SoftUpdate(torch::nn::Module& source,torch::nn::Module& target);


private:
    PolicyNetCont m_actor{ nullptr };
    PolicyNetCont m_targetActor{ nullptr };

    QValueNetCont m_critic{ nullptr };
    QValueNetCont m_targetCritic{ nullptr };

    std::unique_ptr<torch::optim::Adam> m_actorOptimizer;
    std::unique_ptr<torch::optim::Adam> m_criticOptimizer;

  
    int64_t m_stateDim = 0;
    int64_t m_actionDim = 0;

    size_t m_bufferCapacity = 100000;
    size_t m_batchSize = 64;

    double m_actorLearningRate = 1e-4;
    double m_criticLearningRate = 1e-3;
    double m_gamma = 0.99;
    double m_tau = 0.005;
    
    double m_noiseStd = 0.2;
    double m_noiseDecay = 0.9995;

    double m_dbSigma = 0.01;
};