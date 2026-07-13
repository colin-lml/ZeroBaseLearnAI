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
        return torch::multinomial(logits_, 1, true).squeeze(1);
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

int Reinforce::TakeAction(VectorDouble s0, bool bPredict)
{
    torch::NoGradGuard no_grad;
	auto s = VectorDoubleTensor(s0);
	auto logits = m_Qnet->forward(s);
    torch::Tensor action;
    if (bPredict)
    {
        action = logits.argmax(-1);
       // cout <<"a: "<< action.sizes() << endl;
    }
    else
    {
        Categorical categorical(logits);
        action = categorical.sample();
    }

	return action.item<int>();
}

void Reinforce::TrainData(int maxCount)
{
    cout << "Reinforce -> TrainData....." << endl;

    auto adam = CreateOptimizer(m_Qnet);
    m_Qnet->train();

    for (int i = 0; i < maxCount; i++)
    {  
        auto s = m_CartPoleEnv.reset();
        auto done = false;
        int64_t rewardCount = 0;
        VectorRecordDict vList;
        while (!done && rewardCount < 470)
        {
            auto a = TakeAction(s);
            //{ state, reward, terminated, truncated };
            auto [s1, r, b, t] = m_CartPoleEnv.step(a);

            // s,a,r,
            vList.push_back({ s, a, r });

            rewardCount += r;
            done = b;
            s = s1;
           
        }

        Update(adam, vList);

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

    for (int i = 0; i < 10; i++)
    {
        auto s0 = m_CartPoleEnv.reset();
        auto done = false;
        int64_t rewardCount = 0;
        int64_t step = 0;
        while (!done && step < 500)
        {
            auto a = TakeAction(s0, true);
            //{ state, reward, terminated, truncated };
            auto [s1, r, d, _] = m_CartPoleEnv.step(a);
            done = d;
            s0 = s1;
            rewardCount += r;
            step++;
        }
        cout << "count: " << i + 1 << " , rewardCount: " << rewardCount << endl;
    }

}

void Reinforce::Update(torch::optim::Adam& adam, VectorRecordDict& vList)
{
    if (vList.size()==0)
    {
        return;
    }

    int length = vList.size()-1;
    double G = 0;
    adam.zero_grad();

    for (int i = length; 0 <= i; i--)
    {
        // s,a,r,
        auto [s, a, r] = vList[i];
        auto s0 = VectorDoubleTensor(s);
        auto act = torch::tensor({ {a} }, torch::kInt);

        auto action = m_Qnet->forward(s0).gather(1, act);
        auto logprob = torch::log(action + 1e-8);
        G = m_dbGamma * G + r;
        auto lass = -logprob * G;
        lass.backward();
       
    }
    adam.step();
}
