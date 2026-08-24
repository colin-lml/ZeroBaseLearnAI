#include "pch.h"
#include "SAC.h"
#include <chrono>
#include <cstdlib>
#include <unordered_set>


void SAC::GenerateTrainData(int maxCount)
{
    cout << "Currently SAC (continuous)" << endl;
    m_maxMewardCount = 200;
    m_minLogCount = 20;
    m_minLogStep = 6;

    // 超参数（可按需要调整）
    m_dbGamma = 0.98;
    m_dbTau = 0.005;
    m_batchSize = 64;

    GetReplayDataList().clear();

    auto input = m_objEnv->GetStateDim();
    auto output = m_objEnv->GetActionDim();
    auto actionBound = m_objEnv->GetActionHigh();

    TORCH_CHECK(output == 1, "SAC currently supports one-dimensional continuous actions");

    m_actor = SACPolicyNetCont(input, output, actionBound);
    m_critic1 = QValueNetCont(input, output);
    m_critic2 = QValueNetCont(input, output);
    m_targetCritic1 = QValueNetCont(input, output);
    m_targetCritic2 = QValueNetCont(input, output);

    m_actor->to(m_device);
    m_critic1->to(m_device);
    m_critic2->to(m_device);
    m_targetCritic1->to(m_device);
    m_targetCritic2->to(m_device);

    // 将目标网络初始化为 critic 网络参数
    CopyModuleParameters(*m_critic1, *m_targetCritic1);
    CopyModuleParameters(*m_critic2, *m_targetCritic2);

    // 优化器
    m_pActorOpt = std::make_unique<torch::optim::Adam>(m_actor->parameters(), torch::optim::AdamOptions(m_dbActorLRDefault));
    m_pCritic1Opt = std::make_unique<torch::optim::Adam>(m_critic1->parameters(), torch::optim::AdamOptions(m_dbCriticLRDefault));
    m_pCritic2Opt = std::make_unique<torch::optim::Adam>(m_critic2->parameters(), torch::optim::AdamOptions(m_dbCriticLRDefault));

    // 可训练 log alpha (初始化为 log(0.01))
    m_logAlpha = torch::full({}, std::log(0.01), torch::TensorOptions().device(m_device).dtype(torch::kFloat32));
    m_logAlpha.set_requires_grad(true);
    // alpha 优化器，使用单张 tensor 参数列表
    m_pAlphaOpt = std::make_unique<torch::optim::Adam>(std::initializer_list<torch::Tensor>{m_logAlpha}, torch::optim::AdamOptions(m_dbAlphaLRDefault));

    // 训练模式
    m_actor->train();
    m_critic1->train();
    m_critic2->train();
    m_targetCritic1->eval();
    m_targetCritic2->eval();

    // 连续动作 SAC 的目标熵通常为 -动作维度
    m_dbTargetEntropy = -static_cast<double>(m_objEnv->GetActionDim());

    // 使用 BaseAdvanced 统一的数据生成循环（内部会调用 TrainGenerateItem1/2）
    BaseAdvanced::GenerateTrainData(maxCount);

    // eval 模式
    m_actor->eval();
    m_critic1->eval();
    m_critic2->eval();

    // 释放资源（unique_ptr 会自动释放）
    m_pActorOpt.reset();
    m_pCritic1Opt.reset();
    m_pCritic2Opt.reset();
    m_pAlphaOpt.reset();
}

double SAC::TakeAction(VectorDouble& s0, bool bPredict)
{
    torch::NoGradGuard no_grad;
    auto s = VectorDoubleTensor(s0, m_device);
    torch::Tensor action;
    if (bPredict)
    {
        action = m_actor->mean_action(s);
    }
    else
    {
        auto result = m_actor->forward(s);
        action = std::get<0>(result);
    }

    auto value = action.squeeze().item<double>();
    return std::clamp(value, m_objEnv->GetActionLow(), m_objEnv->GetActionHigh());
}

void SAC::TrainGenerateItem1(const QwItem& item)
{
    // 与 DDPG/其它模块一致，先把样本加入全局 replay，然后在达到最小容量后在 Update 中训练
    AddReplayDataList(item);

    // 如果回放池容量大于阈值，进行一次更新
    if (GetReplayDataList().size() > static_cast<size_t>(m_nMinimalsize))
    {
        Update();
    }
}

void SAC::TrainGenerateItem2(const QwList& vList)
{


}

// 软更新 target 网络： target = (1 - tau) * target + tau * source
void SAC::SoftUpdate(torch::nn::Module& source, torch::nn::Module& target)
{
    torch::NoGradGuard no_grad;
    auto srcParams = source.parameters();
    auto tgtParams = target.parameters();
    TORCH_CHECK(srcParams.size() == tgtParams.size(), "SAC::SoftUpdate: parameter count mismatch");

    for (size_t i = 0; i < srcParams.size(); ++i)
    {
        tgtParams[i].mul_(1.0 - m_dbTau);
        tgtParams[i].add_(srcParams[i], m_dbTau);
    }
}

void SAC::Update()
{


    ReplayBuffer replayBuffer;
    auto samples = replayBuffer.sample(m_batchSize);

    if (samples.empty())
    {
        return;
    }

    auto [s0, a, reward, s1, done] = QwListToTensor(samples, m_device, true);

    torch::Tensor tdTarget;

    
    {
        torch::NoGradGuard noGrad;
        const auto alpha = m_logAlpha.exp();
        auto [nextAction, nextLogProb] = m_actor->forward(s1);
        const auto minTargetQ =torch::min(m_targetCritic1->forward(s1, nextAction), m_targetCritic2->forward(s1, nextAction));

        const auto nextValue =minTargetQ - alpha * nextLogProb;

        tdTarget =reward+ m_dbGamma * (1.0 - done) * nextValue;
    }

    // 更新两个Q网络
    {
        auto criticLoss1 = torch::mean(torch::mse_loss(m_critic1->forward(s0, a), tdTarget.detach()));
        auto criticLoss2 = torch::mean(torch::mse_loss(m_critic2->forward(s0, a), tdTarget.detach()));
        m_pCritic1Opt->zero_grad();
        m_pCritic2Opt->zero_grad();
        criticLoss1.backward();
        criticLoss2.backward();
        m_pCritic1Opt->step();
        m_pCritic2Opt->step();
    }
    // 更新策略网络
   
    torch::Tensor detachedLogProb;
    {
        auto [newAction, logProb] = m_actor->forward(s0);
        logProb = -logProb;
        detachedLogProb = logProb.detach();

		auto q1 = m_critic1->forward(s0, newAction);
		auto q2 = m_critic2->forward(s0, newAction);
        auto actorLoss = torch::mean(-m_logAlpha.exp() * logProb - torch::min(q1,q2));
		m_pActorOpt->zero_grad();
        actorLoss.backward();
		m_pActorOpt->step();
    }
    //更新alpha值
    {
    
        auto alphaLoss = torch::mean((detachedLogProb - m_dbTargetEntropy).detach() * m_logAlpha.exp());
        m_pAlphaOpt->zero_grad();
        alphaLoss.backward();
		m_pAlphaOpt->step();
    }


    SoftUpdate(*m_critic1, *m_targetCritic1);
    SoftUpdate(*m_critic2, *m_targetCritic2);
}
