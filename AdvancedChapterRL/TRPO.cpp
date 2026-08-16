#include "pch.h"
#include "TRPO.h"
#include <vector>


torch::Tensor parameters_to_vector(const std::vector<torch::Tensor>& params)
{
    if (params.empty())
    {
        return torch::empty({ 0 });
    }

    // 收集所有非空参数的扁平视图
    std::vector<torch::Tensor> flats;
    flats.reserve(params.size());
    auto first = params.front();
    auto options = torch::TensorOptions().dtype(first.dtype()).device(first.device());

    for (const auto& p : params) 
    {
        if (!p.defined() || p.numel() == 0) 
        {
            continue;
        }
        // 保证连续并扁平化
        flats.push_back(p.contiguous().view({ -1 }));
    }

    if (flats.empty()) {
        return torch::empty({ 0 }, options);
    }

    // 在参数所在 device 上连接
    return torch::cat(flats, 0);
}

torch::Tensor parameters_to_vector(const torch::nn::Module& module)
{
    std::vector<torch::Tensor> params;
    for (const auto& p : module.parameters())
    {
        params.push_back(p);
    }
    return parameters_to_vector(params);
}

void vector_to_parameters(const torch::Tensor& flat, std::vector<torch::Tensor>& params)
{
    // 展平为一维连续张量；若原flat不连续会触发一次拷贝，保证narrow可用
    auto flat_view = flat.contiguous().view({ -1 });
    const int64_t total = flat_view.numel();

    int64_t expected = 0;
    for (const auto& p : params)
    {
        if (!p.defined()) continue;
        const int64_t numel = p.numel();
        if (numel == 0) continue;
        expected += numel;
    }

    TORCH_CHECK(
        expected == total,
        "vector_to_parameters: size mismatch. expected=", expected, ", got=", total
    );

    int64_t offset = 0;
    for (auto& p : params)
    {
        if (!p.defined()) continue;
        const int64_t numel = p.numel();
        if (numel == 0) continue;

        auto slice = flat_view.narrow(0, offset, numel);
        // to(options) 自动包含 device + dtype，无需两次to，内部优化避免冗余拷贝
        auto chunk = slice.to(p.options()).view(p.sizes());
        p.copy_(chunk);

        offset += numel;
    }

    TORCH_CHECK(offset == expected, "vector_to_parameters: internal offset mismatch");
}

void vector_to_parameters(const torch::Tensor& flat, torch::nn::Module& module)
{
    std::vector<torch::Tensor> params;
    for (auto& p : module.parameters()) 
    {
        params.push_back(p);
    }
    vector_to_parameters(flat, params);
}

// 将一个 module 的参数复制到另一个 module（替代 Python 的 load_state_dict）
static void CopyModuleParameters(const torch::nn::Module& src, torch::nn::Module& dst)
{
    torch::NoGradGuard no_grad;

    // 构建 src 参数的查找表： name -> tensor
    std::unordered_map<std::string, torch::Tensor> src_map;
    for (const auto& item : src.named_parameters(/*recurse=*/true))
    {
        src_map.emplace(item.key(), item.value());
    }

    // 遍历 dst 的参数，按 name 查找并 copy_
    for (auto& item : dst.named_parameters(/*recurse=*/true)) 
    {
        const auto& name = item.key();
        auto& dst_tensor = item.value();
        auto it = src_map.find(name);
        if (it == src_map.end())
        {
            // 名称不匹配：跳过（或根据需要记录警告）
            continue;
        }
        auto src_tensor = it->second;
        // 保证类型/设备一致，然后原地复制
        if (src_tensor.device() != dst_tensor.device())
        {
            src_tensor = src_tensor.to(dst_tensor.device());
        }
        if (src_tensor.dtype() != dst_tensor.dtype())
        {
            src_tensor = src_tensor.to(dst_tensor.dtype());
        }
        dst_tensor.copy_(src_tensor);
    }
}



torch::Tensor TRPO::ComputeAdvantage(double gamma, double lmbda, torch::Tensor td_delta)
{
    // Ensure tensor is detached and on CPU for host-side loop
    auto td = td_delta.detach().cpu();
    // Convert to contiguous float tensor
    td = td.contiguous();

    // Expect td shape [n, m] (timesteps, batch or parallel envs)
    auto n = td.size(0);
    auto m = (td.dim() > 1) ? td.size(1) : 1;

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
    auto adv_tensor = torch::from_blob(advantages.data(), { (int64_t)n, (int64_t)m }, options).clone();
    return adv_tensor.to(td_delta.device());
}

torch::Tensor TRPO::ComputeSurrogateObj(torch::Tensor s, torch::Tensor a, torch::Tensor advantage, torch::Tensor logProbs, PolicyNet& actorNet)
{
    // actorNet 的 forward 在 PolicyNetImpl 中返回的是 softmax 概率
    auto probs = actorNet->forward(s); // [B, A]

    // 规范 actions 形状为 [B,1]
    if (a.dim() == 1)
    {
        a = a.unsqueeze(1);
    }

    // 取出当前策略下被采取动作的概率并计算 log-prob
    const double eps = 1e-8;
    auto action_probs = probs.gather(1, a);        // [B,1]
    auto log_probs = torch::log(action_probs + eps);     // [B,1]

    // 规范 advantage 形状为 [B,1]
    if (advantage.dim() == 1) 
    {
        advantage = advantage.unsqueeze(1);
    }

    // 计算比率 r(θ) = exp(logπ_θ(a|s) - logπ_old(a|s))
    auto ratio = torch::exp(log_probs - logProbs);

    // 计算 surrogate objective: mean( r * advantage )
    auto surrogate = (ratio * advantage).mean();

    return surrogate;
}

torch::Tensor TRPO::HessianMatrixVectorProduct(torch::Tensor s, Categorical actionDists, torch::Tensor v)
{
#if 0
    auto newActionDists = Categorical(m_ActorNet->forward(s));
    auto kl = torch::mean(actionDists.kl_divergence(newActionDists));

    // 一阶梯度，保留计算图以便后续二阶导（create_graph = true）
    auto params = std::vector<torch::Tensor>();
    for (auto& p : m_ActorNet->parameters())
    {
        params.push_back(p);
    }

    auto grads = torch::autograd::grad({ kl }, params, {}, true, true, true);
    vector<torch::Tensor> flat;
    for (auto& t : grads)
    {
        flat.push_back(t.view({ -1 }));
    }
    auto vectorGrad = torch::cat(flat);
    auto klGradVectorProduct = torch::dot(vectorGrad, v);

    auto grad2 = torch::autograd::grad({ klGradVectorProduct }, params, {}, true, false,true);
    flat.clear();
    for (auto& t : grads)
    {
        flat.push_back(t.view({ -1 }));
    }

	return torch::cat(flat);   
#endif


    // 计算 KL（batch mean）
    auto newActionDists = Categorical(m_ActorNet->forward(s));
    auto kl = torch::mean(actionDists.kl_divergence(newActionDists));

    // 一阶梯度，保留计算图以便后续二阶导（create_graph = true）
    auto params = std::vector<torch::Tensor>();
    for (auto& p : m_ActorNet->parameters()) params.push_back(p);

    auto grads = torch::autograd::grad({ kl }, params, /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/true, /*allow_unused=*/true);

    // 将 grads 扁平化为向量 (与参数顺序一致)，未定义时填 0
    std::vector<torch::Tensor> flat_grad_parts;
    flat_grad_parts.reserve(grads.size());
    for (size_t i = 0; i < params.size(); ++i)
    {
        const auto& p = params[i];
        auto g = grads[i];
        if (!g.defined())
        {
            // 填零，类型/设备与参数一致
            flat_grad_parts.push_back(torch::zeros({ p.numel() }, torch::TensorOptions().dtype(p.dtype()).device(p.device())));
        }
        else
        {
            flat_grad_parts.push_back(g.contiguous().view({ -1 }));
        }
    }

    auto vectorGrad = torch::cat(flat_grad_parts).to(v.device()).to(v.dtype());

    // 内积得到标量
    auto klGradVectorProduct = torch::dot(vectorGrad, v);

    // 对该标量再对参数求导，得到 Hessian-vector 乘积（允许缺失项）
    auto grad2 = torch::autograd::grad({ klGradVectorProduct }, params, /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/false, /*allow_unused=*/true);

    // 扁平化 grad2，缺失处填 0
    std::vector<torch::Tensor> flat2_parts;
    flat2_parts.reserve(grad2.size());
    for (size_t i = 0; i < params.size(); ++i)
    {
        const auto& p = params[i];
        auto g2 = grad2[i];
        if (!g2.defined())
        {
            flat2_parts.push_back(torch::zeros({ p.numel() }, torch::TensorOptions().dtype(p.dtype()).device(p.device())));
        }
        else
        {
            flat2_parts.push_back(g2.contiguous().view({ -1 }));
        }
    }

    auto Hv = torch::cat(flat2_parts).to(v.device()).to(v.dtype());
    return Hv;
}


torch::Tensor TRPO::ConjugateGradient(torch::Tensor objGrad, torch::Tensor s, Categorical actionDists)
{
    auto x = torch::zeros_like(objGrad);
	auto r = objGrad.clone();
	auto p = r.clone();
	auto rdotr = torch::dot(r, r);
    for (int i = 0; i < 10; i++)
    {
	  auto Hp = HessianMatrixVectorProduct(s, actionDists, p);
	  auto alpha = rdotr / (torch::dot(p, Hp) + 1e-8);
	  x += alpha * p;
	  r -= alpha * Hp;
	  auto new_rdotr = torch::dot(r, r);
      if (new_rdotr.item<double>() < 1e-9)
      {
          break;
      }
	  auto beta = new_rdotr / (rdotr + 1e-8);
	  p = r + beta * p;
	  rdotr = new_rdotr;

    }

	return x; 
}

torch::Tensor TRPO::LineSearch(torch::Tensor s, torch::Tensor a, torch::Tensor advantage, torch::Tensor logProbs, Categorical actionDists, torch::Tensor step_size)
{
#if 0
   auto oldData = parameters_to_vector(m_ActorNet->parameters());
   auto oldSurrogate = ComputeSurrogateObj(s, a, advantage, logProbs, m_ActorNet);
   auto input = m_CartPoleEnv.GetStateDim();
   auto output = m_CartPoleEnv.GetActionDim();

   auto tmepActor = PolicyNet(input, output);
   tmepActor->to(m_device);
   

   CopyModuleParameters(*m_ActorNet, *tmepActor);

   for (int i = 0; i < 15; i++)
   {
      auto coef = std::pow(m_dbAlpha, i);
	  auto newParams = oldData + coef * step_size;

      vector_to_parameters(newParams, *tmepActor);
	  auto newActionDists = Categorical(tmepActor->forward(s));
	  torch::Tensor kl = torch::mean(actionDists.kl_divergence(newActionDists));
      auto newSurrogate = ComputeSurrogateObj(s, a, advantage, logProbs, tmepActor);
      if (newSurrogate.item<double>() > oldSurrogate.item<double>() &&  kl.item<double>() > m_dbklConstraint)
      {
          return newParams;
	  }
   }

   return oldData;
#endif

   auto oldData = parameters_to_vector(m_ActorNet->parameters());
   auto oldSurrogate = ComputeSurrogateObj(s, a, advantage, logProbs, m_ActorNet);
   auto input = m_CartPoleEnv.GetStateDim();
   auto output = m_CartPoleEnv.GetActionDim();

   auto tmpActor = PolicyNet(input, output);
   tmpActor->to(m_device);

   CopyModuleParameters(*m_ActorNet, *tmpActor);

   for (int i = 0; i < 15; i++)
   {
       auto coef = std::pow(m_dbAlpha, i);
       auto newParams = oldData + coef * step_size;
       
       auto params = std::vector<torch::Tensor>();
       for (auto& p : m_ActorNet->parameters())
       {
           params.push_back(p);
       }
       vector_to_parameters(newParams, params);

       auto newActionDists = Categorical(tmpActor->forward(s));
       torch::Tensor kl = torch::mean(actionDists.kl_divergence(newActionDists));
       auto newSurrogate = ComputeSurrogateObj(s, a, advantage, logProbs, tmpActor);

       // 接受新参数：surrogate 提升且 KL 在约束之内（注意 <）
       if (newSurrogate.item<double>() > oldSurrogate.item<double>() && kl.item<double>() < m_dbklConstraint)
       {
           return newParams;
       }
   }

   return oldData;

}

void TRPO::PolicyLearnUpdate(torch::Tensor s, torch::Tensor a, Categorical actionDists, torch::Tensor logProbs, torch::Tensor advantage)
{
    auto surrogate_obj = ComputeSurrogateObj(s,a,advantage, logProbs,m_ActorNet);
	auto grads = torch::autograd::grad({ surrogate_obj }, m_ActorNet->parameters());
	vector<torch::Tensor> flat;
    for (auto& t : grads) 
    {
        flat.push_back(t.view({ -1 })); 
    }
    auto flatGrad = torch::cat(flat);
    auto descent_direction = ConjugateGradient(flatGrad, s, actionDists);
	auto Hd = HessianMatrixVectorProduct(s, actionDists, descent_direction);
	auto step_size = torch::sqrt(2 * m_dbklConstraint / (torch::dot(descent_direction, Hd) + 1e-8));
    auto new_para = LineSearch(s, a, advantage, logProbs, actionDists, step_size);
	vector_to_parameters(new_para, *m_ActorNet);
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
    m_CriticNet = ValueNet(input, output);

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

    auto [s0, a, r, s1, done] = QwListToTensor(vList, m_device);

    auto v0 = m_CriticNet->forward(s0);
    auto v1 = r + m_dbGamma * m_CriticNet->forward(s1) * (1 - done);

    auto td = v1 - v0;
    auto advantage = ComputeAdvantage(m_dbGamma, m_dbLmbda,td);

    auto action = m_ActorNet->forward(s0).gather(1, a);
    auto logProbs = torch::log(action + 1e-8).detach();
    auto actionDists = Categorical(m_ActorNet->forward(s0).detach());

    auto criticLoss = torch::mean(torch::mse_loss(v0, v1.detach()));

    m_pAdamCritic->zero_grad();

    criticLoss.backward();
    m_pAdamCritic->step();

    PolicyLearnUpdate(s0,a, actionDists,logProbs,advantage);

}

