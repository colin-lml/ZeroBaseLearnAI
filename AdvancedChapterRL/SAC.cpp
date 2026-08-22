#include "pch.h"
#include "SAC.h"


void SAC::GenerateTrainData(int maxCount)
{
    cout << "Currently SAC (discrete)" << endl;

    // 超参数（可按需要调整）
    m_dbGamma = 0.98;
    m_dbTau = 0.005;
    m_batchSize = 64;

    auto input = m_objEnv->GetStateDim();
    auto output = m_objEnv->GetActionDim();

    // actor: 使用已有 PolicyNet（返回概率）
    m_actor = PolicyNet(input, output);
    m_critic1 = DQNQnet(input, output);
    m_critic2 = DQNQnet(input, output);
    m_targetCritic1 = DQNQnet(input, output);
    m_targetCritic2 = DQNQnet(input, output);

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

    // 目标熵按动作维度设置（离散）: -|A|
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
    auto probs = m_actor->forward(s); // [B, A] 这里 B=1
    torch::Tensor action;
    if (bPredict)
    {
        action = probs.argmax(-1);
    }
    else
    {
        Categorical categorical(probs);
        action = categorical.sample();
    }

    return action.item<int>();
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
    // 从全局回放中采样
    ReplayBuffer dataTrain;
    auto samples = dataTrain.sample(m_batchSize);
    auto [s0, a, r, s1, done] = QwListToTensor(samples, m_device, true);

    // ------------------ 计算目标 Q 值 ------------------
    torch::Tensor td_target;
    {
        torch::NoGradGuard no_grad;

        // next action probabilities & entropy
        auto next_probs = m_actor->forward(s1); // [B, A]
        auto next_log_probs = torch::log(next_probs + 1e-8);
        auto next_entropy = - (next_probs * next_log_probs).sum(-1, true); // [B,1]

        // target q values
        auto q1_next = m_targetCritic1->forward(s1); // [B, A]
        auto q2_next = m_targetCritic2->forward(s1); // [B, A]
        auto min_q_next = torch::min(q1_next, q2_next); // [B, A]

        // 期望下的下个状态值：sum_a pi(a|s') * min_q + alpha * entropy
        auto expected_min_q = (next_probs * min_q_next).sum(-1, true); // [B,1]
        auto alpha = m_logAlpha.exp();
        auto next_value = expected_min_q + alpha * next_entropy; // [B,1]

        td_target = r + m_dbGamma * next_value * (1.0 - done); // [B,1]
    }

    // ------------------ Critic 更新 ------------------
    // critic1
    {
        auto q1 = m_critic1->forward(s0).gather(1, a); // [B,1]
        auto loss1 = torch::mse_loss(q1, td_target.detach());
        m_pCritic1Opt->zero_grad();
        loss1.backward();
        m_pCritic1Opt->step();
    }

    // critic2
    {
        auto q2 = m_critic2->forward(s0).gather(1, a); // [B,1]
        auto loss2 = torch::mse_loss(q2, td_target.detach());
        m_pCritic2Opt->zero_grad();
        loss2.backward();
        m_pCritic2Opt->step();
    }

    // ------------------ Actor 更新 ------------------
    {
        auto probs = m_actor->forward(s0); // [B, A]
        auto log_probs = torch::log(probs + 1e-8);
        auto entropy = - (probs * log_probs).sum(-1, true); // [B,1]

        auto q1 = m_critic1->forward(s0); // [B, A]
        auto q2 = m_critic2->forward(s0); // [B, A]
        auto min_q = torch::min(q1, q2); // [B, A]

        // 期望 Q： E_{a~pi}[ min_q(s,a) ]
        auto expected_min_q = (probs * min_q).sum(-1, true); // [B,1]

        auto alpha = m_logAlpha.exp();

        // actor loss: minimize -(expected_min_q + alpha * entropy)  <=> maximize expected_min_q + alpha * entropy
        auto actor_loss = (-(expected_min_q + alpha * entropy)).mean();

        m_pActorOpt->zero_grad();
        actor_loss.backward();
        m_pActorOpt->step();
    }

    // ------------------ alpha 自适应 ------------------
    {
        auto probs = m_actor->forward(s0);
        auto log_probs = torch::log(probs + 1e-8);
        auto entropy = - (probs * log_probs).sum(-1, true); // [B,1]
        // alpha loss: -log_alpha * (entropy + target_entropy).detach()
        auto alpha_loss = ( - m_logAlpha * ( (entropy + m_dbTargetEntropy).detach() ) ).mean();
        m_pAlphaOpt->zero_grad();
        alpha_loss.backward();
        m_pAlphaOpt->step();
    }

    // ------------------ 软更新 target networks ------------------
    SoftUpdate(*m_critic1, *m_targetCritic1);
    SoftUpdate(*m_critic2, *m_targetCritic2);
}
