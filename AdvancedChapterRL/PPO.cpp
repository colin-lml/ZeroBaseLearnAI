#include "pch.h"
#include "PPO.h"


torch::Tensor PPO::ComputeAdvantage(double gamma, double lmbda, torch::Tensor& td)
{
    auto device = td.device();
    td.detach_();
    // Ensure tensor is detached and on CPU for host-side loop
    td = td.cpu().contiguous();

    // Expect td shape [n, m] (timesteps, batch or parallel envs)
    auto n = td.size(0);
    auto m = td.size(1);

    std::vector<float> advantages(static_cast<size_t>(n * m));

    // Compute advantage separately for each column (second dimension)
    for (int64_t col = 0; col < m; ++col)
    {
        double adv = 0.0;
        for (int64_t i = n - 1; i >= 0; --i)
        {
            // td[i][col] should be a scalar tensor
            double delta = (m == 1) ? td[i].item<double>() : td[i][col].item<double>();
            adv = gamma * lmbda * adv + delta;
            advantages[static_cast<size_t>(i * m + col)] = static_cast<float>(adv);
        }
    }

    // Create tensor on same device as original td_delta with shape [n, m]
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto adv = torch::from_blob(advantages.data(), { (int64_t)n, (int64_t)m }, options).clone();
    return adv.to(device);

}




double PPO::TakeAction(VectorDouble& s0, bool bPredict)
{
    torch::NoGradGuard no_grad;
    auto s = VectorDoubleTensor(s0,m_device);
    auto logits = m_ActorNet->forward(s);
    torch::Tensor action;
    if (bPredict)
    {
        action = logits.argmax(-1);
    }
    else
    {
        Categorical categorical(logits);
        action = categorical.sample();
    }

    return action.item<int>();
}

void PPO::GenerateTrainData(int maxCount)
{
    cout << "Currently PPO" << endl;

    m_dbGamma = 0.98;
   

    auto input = m_objEnv->GetStateDim();
    auto output = m_objEnv->GetActionDim();

    m_ActorNet = PolicyNet(input, output);
    m_CriticNet = ValueNet(input, 1);

    m_CriticNet->to(m_device);
    m_ActorNet->to(m_device);

    m_pAdamActor = new torch::optim::Adam(m_ActorNet->parameters(), { m_dbActorLR });
    m_pAdamCritic = new torch::optim::Adam(m_CriticNet->parameters(), { m_dbCriticLR });


    m_ActorNet->train();
    m_CriticNet->train();

    BaseAdvanced::GenerateTrainData(maxCount);

    m_ActorNet->eval();
    m_CriticNet->eval();
    delete m_pAdamActor;
    m_pAdamActor = nullptr;

    delete m_pAdamCritic;
    m_pAdamCritic = nullptr;

}

void PPO::TrainGenerateItem2(const QwList& vList)
{

    static int count = 0;

    if (450 < vList.size())
    {
        count++;
        if (3 < count)
        {
            m_bEndGenerateTrain = true;
            return;
        }
    }
    else
    {
        count = 0;
    }

    auto [s0, a, r, s1, done] = QwListToTensor(vList, m_device);

    auto v0 = m_CriticNet->forward(s0);
    auto v1 = r + m_dbGamma * m_CriticNet->forward(s1).detach() * (1 - done);
    auto td = v1 - v0;

    auto criticLoss = torch::mean(torch::mse_loss(v0, v1.detach()));
    m_pAdamCritic->zero_grad();
    criticLoss.backward();
    m_pAdamCritic->step();

    auto adv = ComputeAdvantage(m_dbGamma, m_dbLmbda, td);

    //防止 advantage 数值爆炸 / 剧烈波动, 优势归一化
    auto mean = adv.mean();
    auto std = adv.std();
    // +1e‑8 防止std为0除零
    auto adv_norm = ((adv - mean) / (std + 1e-8)).detach();

    auto oldLogProbs = torch::log(m_ActorNet->forward(s0).gather(1, a)).detach();
    
    for (int i = 0; i < m_nPPOEpochs; i++)
    {
         auto  logProbs = torch::log(m_ActorNet->forward(s0).gather(1, a));
         auto  ratio = torch::exp(logProbs - oldLogProbs);
         auto  surr1 = ratio * adv_norm;
		 auto  surr2 = torch::clamp(ratio, 1.0 - m_dbEps, 1.0 + m_dbEps) * adv_norm; // 截断
         auto actorLoss = torch::mean(-torch::min(surr1, surr2));  // # PPO损失函数
         
         m_pAdamActor->zero_grad(); 
         actorLoss.backward();
         m_pAdamActor->step();
    }


}

