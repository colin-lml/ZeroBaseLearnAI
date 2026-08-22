
#pragma once


class SAC : public BaseAdvanced
{
public:
    SAC():BaseAdvanced(true) {}
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
    PolicyNet m_actor;
    DQNQnet m_critic1;
    DQNQnet m_critic2;
    DQNQnet m_targetCritic1;
    DQNQnet m_targetCritic2;

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