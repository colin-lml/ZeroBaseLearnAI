#include "pch.h"
#include "Reinforce.h"

torch::Tensor VectorDoubleTensor(const VectorDouble& item);



class Categorical 
{
public:
    // 构造：仅支持 logits 输入 \[B, K\]
    explicit Categorical(const torch::Tensor& logits)
        : logits_(logits)
    {
        TORCH_CHECK(logits_.dim() == 2, "logits must be shape \[B, num\_actions\]");
    }

    // 采样 action, output shape \[B\], dtype long
    torch::Tensor sample() const
    {
        // Gumbel-Max trick
        auto u = torch::rand_like(logits_);
        auto g = -torch::log(-torch::log(u + 1e-8) + 1e-8);
        auto [_, action] = (logits_ + g).max(1);
        return action;
    }

    // action shape \[B\] long
    // return log\_prob shape \[B\]
    torch::Tensor log_prob(const torch::Tensor& action) const
    {
        auto log_probs = logits_.log_softmax(1);
        auto act_2d = action.unsqueeze(1);       // \[B,1\]
        auto lp = log_probs.gather(1, act_2d);   // \[B,1\]
        return lp.squeeze(1);
    }

   
    torch::Tensor entropy() const
    {
        auto log_probs = logits_.log_softmax(1);
        auto probs = log_probs.exp();
        return -(probs  * log_probs).sum(1);
    }

private:
    torch::Tensor logits_;
};



void Reinforce::PlayCartPole(int maxCount)
{
	torch::manual_seed(32);

	TrainData(maxCount);
}

torch::optim::Adam Reinforce::CreateOptimizer(PolicyNet& model)
{
	torch::optim::AdamOptions opt(m_dbLR);
	opt.betas({ 0.9, 0.98 });
	opt.eps(1e-9);
	opt.weight_decay(0);

	return torch::optim::Adam(model->parameters(), opt);
}

int Reinforce::TakeAction(VectorDouble s0)
{
	auto s = VectorDoubleTensor(s0);
	auto probs = m_Qnet->forward(s);

    Categorical categorical(probs);
    auto action = categorical.sample();

	return action.item<int>();
}

void Reinforce::TrainData(int maxCount)
{
    cout << "Reinforce -> TrainData....." << endl;
    auto adam = CreateOptimizer(m_Qnet);

    for (int i = 0; i < maxCount; i++)
    {  
        auto s = m_CartPoleEnv.reset();
        auto done = false;
        int64_t rewardCount = 0;

        while (!done && rewardCount < 470)
        {
            auto a = TakeAction(s);
            //{ state, reward, terminated, truncated };
            auto [s1, r, b, t] = m_CartPoleEnv.step(a);

            rewardCount += r;
            done = b;
            s = s1;
        }

        if (i % 10 == 0)
        {
            cout << "train i: " << i << " / " << maxCount << " , rewardCount: " << rewardCount << endl;
        }
    }

    TestData();
}

void Reinforce::TestData()
{
    cout << "TestData ....." << endl;

    m_Qnet->eval();

    auto s0 = m_CartPoleEnv.reset();
    auto done = false;
    int64_t rewardCount = 0;
    int64_t step = 0;
    while (!done && step < 500)
    {
        auto a = TakeAction(s0);
        //{ state, reward, terminated, truncated };
        auto [s1, r, d, _] = m_CartPoleEnv.step(a);
        done = d;
        s0 = s1;
        rewardCount += r;
        step++;
    }
    cout << "rewardCount: " << rewardCount << endl;
}