#include "pch.h"
#include "TRPO.h"
#include <vector>



torch::Tensor TRPO::ComputeAdvantage(double gamma, double lmbda, torch::Tensor& td)
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

torch::Tensor TRPO::ComputeSurrogateObj(const torch::Tensor& s, const torch::Tensor& a, const torch::Tensor& adv, const torch::Tensor& oldLogProbs, PolicyNet& actorNet)
{
    
    // actorNet 的 forward 在 PolicyNetImpl 中返回的是 softmax 概率
    auto probs = actorNet->forward(s).gather(1, a); 

    // 取出当前策略下被采取动作的概率并计算 log-prob
   
    auto logProbs = torch::log(probs);     // [B,1]

    // 计算比率 r(θ) = exp(logπ_θ(a|s) - logπ_old(a|s))
    auto ratio = torch::exp(logProbs - oldLogProbs);

    // 计算 surrogate objective: mean( r * advantage )

    return (ratio * adv).mean();
}

torch::Tensor TRPO::HessianMatrixVectorProduct(const torch::Tensor& s, const Categorical& oldsDists, const torch::Tensor& v)
{

    // 计算 KL（batch mean）
    auto newDists = Categorical(m_ActorNet->forward(s));
    auto kl = torch::mean(oldsDists.kl_divergence(newDists));

    // 一阶梯度，保留计算图以便后续二阶导（create_graph = true）
    auto params = std::vector<torch::Tensor>();
    for (auto& p : m_ActorNet->parameters())
    {
        params.push_back(p);
    }

    auto grads = torch::autograd::grad({ kl }, params,{}, true, true, true);

    // 将 grads 扁平化为向量 (与参数顺序一致)，未定义时填 0
    std::vector<torch::Tensor> flatGradParts;
    flatGradParts.reserve(grads.size());

    for (auto& p: grads)
    {
        flatGradParts.push_back(p.contiguous().view({ -1 }));
    }

    auto vectorGrad = torch::cat(flatGradParts).to(v.device()).to(v.dtype());

    // 内积得到标量
    auto klGradVectorProduct = torch::dot(vectorGrad, v);

    // 对该标量再对参数求导，得到 Hessian-vector 乘积（允许缺失项）
    auto grad2 = torch::autograd::grad({ klGradVectorProduct }, params, {}, true, false, true);

    // 扁平化 grad2，缺失处填 0
    std::vector<torch::Tensor> flat2Parts;
    flat2Parts.reserve(grad2.size());

    for (auto& p: grad2)
    {
        flat2Parts.push_back(p.contiguous().view({ -1 }));
    }

    auto Hv = torch::cat(flat2Parts).to(v.device()).to(v.dtype());
    //给 Hessian-vector product 加阻尼
    constexpr double damping = 0.1;
    return Hv + +damping * v;
}


torch::Tensor TRPO::ConjugateGradient(const torch::Tensor& objGrad,const torch::Tensor& s,const Categorical& oldsDists)
{
    auto x = torch::zeros_like(objGrad);
	auto r = objGrad.clone();
	auto p = r.clone();
	auto rdotr = torch::dot(r, r);
    for (int i = 0; i < 20; i++)
    {
	  auto Hp = HessianMatrixVectorProduct(s, oldsDists, p);
	  auto alpha = rdotr / (torch::dot(p, Hp));
	  x += alpha * p;
	  r -= alpha * Hp;
	  auto new_rdotr = torch::dot(r, r);
      if (new_rdotr.item<double>() < 1e-9)
      {
          break;
      }
	  auto beta = new_rdotr / (rdotr);
	  p = r + beta * p;
	  rdotr = new_rdotr;

    }

	return x; 
}

torch::Tensor TRPO::LineSearch(const torch::Tensor& s, const torch::Tensor& a, const torch::Tensor& adv, const torch::Tensor& oldLogProbs, const Categorical& oldsDists, const torch::Tensor& fullStep,bool& bUpdate)
{
   bUpdate = false;
   auto oldParam = ParametersToVector(*m_ActorNet);
   auto oldSurrogate = ComputeSurrogateObj(s, a, adv, oldLogProbs, m_ActorNet);
   auto input = m_CartPoleEnv.GetStateDim();
   auto output = m_CartPoleEnv.GetActionDim();

   auto tmpActor = PolicyNet(input, output);
   tmpActor->to(m_device);
 
   CopyModuleParameters(*m_ActorNet, *tmpActor);

   for (int i = 0; i < 15; i++)
   {
       auto coefficient = std::pow(m_dbAlpha, i);
       auto newParams = oldParam + coefficient * fullStep;
       
       VectorToParameters(newParams, *tmpActor);

       auto newActionDists = Categorical(tmpActor->forward(s));
       torch::Tensor kl = torch::mean(oldsDists.kl_divergence(newActionDists));
       auto newSurrogate = ComputeSurrogateObj(s, a, adv, oldLogProbs, tmpActor);

       auto dbKl = kl.item<double>();

       if (newSurrogate.item<double>() > oldSurrogate.item<double>() && dbKl < m_dbklConstraint)
       {
           bUpdate = true;
           return newParams.detach();
       }
   }

   return oldParam;

}

void TRPO::PolicyLearnUpdate(const torch::Tensor& s, const torch::Tensor& a, const Categorical& oldsDists, const torch::Tensor& oldLogProbs, const torch::Tensor& adv)
{
    auto surrogateObj = ComputeSurrogateObj(s,a, adv, oldLogProbs,m_ActorNet);
	auto grads = torch::autograd::grad({ surrogateObj }, m_ActorNet->parameters());
	vector<torch::Tensor> flat;
    for (auto& t : grads) 
    {
        flat.push_back(t.view({ -1 })); 
    }
    auto flatGrad = torch::cat(flat);
    auto searchDirection = ConjugateGradient(flatGrad, s, oldsDists);
	auto Hd = HessianMatrixVectorProduct(s, oldsDists, searchDirection);
	auto stepScale = torch::sqrt(2 * m_dbklConstraint / (torch::dot(searchDirection, Hd)));
    bool bUpdate;
    auto fullStep = (stepScale * searchDirection).detach();
    auto new_para = LineSearch(s, a, adv, oldLogProbs, oldsDists, fullStep, bUpdate);
    if (bUpdate)
    {
        VectorToParameters(new_para, *m_ActorNet);
    }
	
}



int TRPO::TakeAction(VectorDouble& s0, bool bPredict)
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

void TRPO::GenerateTrainData(int maxCount)
{
    cout << "Currently TRPO" << endl;

    m_dbGamma = 0.98;
    m_dbAlpha = 0.5;

    auto input = m_CartPoleEnv.GetStateDim();
    auto output = m_CartPoleEnv.GetActionDim();

    m_ActorNet = PolicyNet(input, output);
    m_CriticNet = ValueNet(input, 1);

    m_CriticNet->to(m_device);
    m_ActorNet->to(m_device);

   
    m_pAdamCritic = new torch::optim::Adam(m_CriticNet->parameters(), { m_dbCriticLR });


    m_ActorNet->train();
    m_CriticNet->train();

    BaseAdvanced::GenerateTrainData(maxCount);

    m_ActorNet->eval();
    m_CriticNet->eval();


    delete m_pAdamCritic;
    m_pAdamCritic = nullptr;

}

void TRPO::TrainGenerateItem2(const QwList& vList)
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
    auto v1 = r + m_dbGamma * m_CriticNet->forward(s1) * (1 - done);
    auto td = v1 - v0;
    auto criticLoss = torch::mean(torch::mse_loss(v0, v1.detach()));
    m_pAdamCritic->zero_grad();
    criticLoss.backward();
    m_pAdamCritic->step();

    auto adv = ComputeAdvantage(m_dbGamma, m_dbLmbda, td);

    auto logProbs = torch::log(m_ActorNet->forward(s0).gather(1, a)).detach();
    auto actionDists = Categorical(m_ActorNet->forward(s0).detach());
    /// 
    //PolicyLearnUpdate(s0,a, actionDists,logProbs, adv);

    //防止 advantage 数值爆炸 / 剧烈波动, 优势归一化
    auto mean = adv.mean();
    auto std = adv.std();
    // +1e‑8 防止std为0除零
    auto adv_norm = (adv - mean) / (std + 1e-8);
    PolicyLearnUpdate(s0, a, actionDists, logProbs, adv_norm);

}

