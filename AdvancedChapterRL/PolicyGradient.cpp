#include "pch.h"
#include "PolicyGradient.h"


void PolicyGradient::CreateOptimizer(PolicyNet& model)
{
	torch::optim::AdamOptions opt(m_dbLR);
	opt.betas({ 0.9, 0.98 });
	opt.eps(1e-9);
	opt.weight_decay(0);

    m_pAdam = new torch::optim::Adam(model->parameters(), opt);
}

void PolicyGradient::GenerateTrainData(int maxCount)
{
    cout << "Currently PolicyGradient" << endl;

    m_dbGamma = 0.98;
    m_dbLR = 1e-3;

    auto input = m_CartPoleEnv.GetStateDim();
    auto output = m_CartPoleEnv.GetActionDim();

    m_Qnet = PolicyNet(input, output);
    m_Qnet->to(m_device);

    CreateOptimizer(m_Qnet);

    m_Qnet->train();
   

    BaseAdvanced::GenerateTrainData(maxCount);

    m_Qnet->eval();
   
    delete m_pAdam;
    m_pAdam = nullptr;
}

int PolicyGradient::TakeAction(VectorDouble& s0, bool bPredict)
{
    torch::NoGradGuard no_grad;
	auto s = VectorDoubleTensor(s0, m_device);
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

void PolicyGradient::TrainGenerateItem2(const QwList& vList)
{
    if (vList.size() == 0)
    {
        return;
    }

    int length = vList.size() - 1;
    double G = 0;
    m_pAdam->zero_grad();

    for (int i = length; 0 <= i; i--)
    {
        // s,a,r,
        auto [s, a, r, s1, d] = vList[i];
        auto s0 = VectorDoubleTensor(s, m_device);
        auto act = torch::tensor({ {a} }, torch::kInt).to(m_device);

        auto action = m_Qnet->forward(s0).gather(1, act);
        auto logprob = torch::log(action + 1e-8);
        G = m_dbGamma * G + r;
        auto lass = -logprob * G;
        lass.backward();

    }
    m_pAdam->step();
}

