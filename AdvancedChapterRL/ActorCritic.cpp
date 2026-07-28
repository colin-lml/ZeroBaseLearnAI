#include "pch.h"
#include "ActorCritic.h"



int ActorCritic::TakeAction(VectorDouble& s0, bool bPredict)
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

void ActorCritic::GenerateTrainData(int maxCount)
{
    cout << "Currently Actor-Critic" << endl;

    m_dbGamma = 0.98;
   

    auto input = m_CartPoleEnv.GetStateDim();
    auto output = m_CartPoleEnv.GetActionDim();

    m_ActorNet = PolicyNet(input, output);
    m_CriticNet = ValueNet(input, output);

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

void ActorCritic::TrainGenerateItem2(const QwList& vList)
{

    auto [s0, a, r, s1, done] = QwListToTensor(vList, m_device);

    auto v0 = m_CriticNet->forward(s0);
    auto v1 = r + m_dbGamma * m_CriticNet->forward(s1) * (1 - done);

    auto td = v1 - v0;

    auto action = m_ActorNet->forward(s0).gather(1, a);
    auto logProbs = torch::log(action + 1e-8);
    auto actorLoss = torch::mean(-logProbs * td.detach());

    auto criticLoss = torch::mean(torch::mse_loss(v0, v1.detach()));

    m_pAdamActor->zero_grad();
    m_pAdamCritic->zero_grad();

    actorLoss.backward();
    criticLoss.backward();


    m_pAdamActor->step();
    m_pAdamCritic->step();
}

