
#pragma once


class NormalDistribution
{
public:
    NormalDistribution(torch::Tensor mean, torch::Tensor std)
        : m_mean(std::move(mean)),m_std(std::move(std))
    {
    }

    // 对应 Python：dist.rsample()
    torch::Tensor rsample() const
    {
        const auto epsilon = torch::randn_like(m_std);
        return m_mean + m_std * epsilon;
    }

    // 对应 Python：dist.log_prob(value)
    torch::Tensor log_prob(const torch::Tensor& value) const
    {
        double logTwoPi = std::log(2 * M_PI);///1.8378770664093453;//

        return -0.5 * ((value - m_mean) / m_std).pow(2)- torch::log(m_std)- 0.5 * logTwoPi;
    }

private:
    torch::Tensor m_mean;
    torch::Tensor m_std;
};



class SACPolicyNetContImpl : public torch::nn::Module
{
public:

    SACPolicyNetContImpl() = default;
    SACPolicyNetContImpl(int64_t input, int64_t output, double actionBound, int64_t hidden = 128)
    {
        m_fc1 = register_module("fc1", torch::nn::Linear(input, hidden));
        m_mu = register_module("mu", torch::nn::Linear(hidden, output));
        m_std = register_module("std", torch::nn::Linear(hidden, output));
        m_dbActionBound = actionBound;
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
        auto mu = m_mu->forward(x);
        auto std = torch::softplus(m_std->forward(x)) + 1e-6;
        
        NormalDistribution normal(mu, std);
        auto normalSample = normal.rsample();
        auto logProb = normal.log_prob(normalSample);
        auto action = torch::tanh(normalSample);

        logProb = logProb - torch::log(1.0 - action.pow(2) + 1e-7);
        logProb = logProb.sum(-1, true);
        action = action * m_dbActionBound;

        return { action, logProb };
    }

    torch::Tensor mean_action(torch::Tensor x)
    {
        x = torch::relu(m_fc1->forward(x));
        return m_dbActionBound * torch::tanh(m_mu->forward(x));
    }

private:
    torch::nn::Linear m_fc1{ nullptr };
    torch::nn::Linear m_mu{ nullptr };
    torch::nn::Linear m_std{ nullptr };
    double m_dbActionBound = 2.0;
};

TORCH_MODULE(SACPolicyNetCont);



class SAC : public BaseAdvanced
{
public:
    SAC():BaseAdvanced(false) {}
    ~SAC() override = default;

protected:
    void GenerateTrainData(int maxCount) override;
    double TakeAction(VectorDouble& s0, bool bPredict = false) override;

    void TrainGenerateItem1(const QwItem& item) override;
    void TrainGenerateItem2(const QwList& vList) override;

private:
    void Update();
    void SoftUpdate(torch::nn::Module& source, torch::nn::Module& target);

    // 网络
    SACPolicyNetCont m_actor;
    QValueNetCont m_critic1;
    QValueNetCont m_critic2;
    QValueNetCont m_targetCritic1;
    QValueNetCont m_targetCritic2;

    // 优化器
    std::unique_ptr<torch::optim::Adam> m_pActorOpt;
    std::unique_ptr<torch::optim::Adam> m_pCritic1Opt;
    std::unique_ptr<torch::optim::Adam> m_pCritic2Opt;
    std::unique_ptr<torch::optim::Adam> m_pAlphaOpt;

    // 可训练的 log alpha
    torch::Tensor m_logAlpha;

    // 超参数（默认值可在 GenerateTrainData 中调整）
    const double m_dbActorLRDefault = 1e-3;
    const double m_dbCriticLRDefault = 1e-3;
    const double m_dbAlphaLRDefault = 1e-3;
    double m_dbGamma = 0.98;
    double m_dbTau = 0.005;
    double m_dbTargetEntropy = -1.0;

    // replay / batch
    const int m_nMinimalsize = 500;
    const int64_t m_batchSizeDefault = 64;
    int64_t m_batchSize = m_batchSizeDefault;

};