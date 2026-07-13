#include "pch.h"
#include "ActorCritic.h"
torch::Tensor VectorDoubleTensor(const VectorDouble& item);


void ActorCritic::PlayCartPole(int maxCount)
{
    torch::manual_seed(42);

    TrainData(maxCount);
}


int ActorCritic::TakeAction(VectorDouble s0, bool bPredict)
{
    torch::NoGradGuard no_grad;
    auto s = VectorDoubleTensor(s0);
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

void ActorCritic::TrainData(int maxCount)
{
    cout << "ActorCritic -> TrainData....." << endl;

    torch::optim::AdamOptions optA(m_dbActorLR);
    torch::optim::AdamOptions optC(m_dbCriticLR);

    torch::optim::Adam actorAdam = torch::optim::Adam(m_ActorNet->parameters(), optA);
    torch::optim::Adam criticAdam = torch::optim::Adam(m_CriticNet->parameters(), optC);
    

    m_ActorNet->train();
    m_CriticNet->train();

    for (int i = 0; i < maxCount; i++)
    {
        auto s = m_CartPoleEnv.reset();
        auto done = false;
        int64_t rewardCount = 0;
        QwList vList;
        while (!done && rewardCount < 470)
        {
            auto a = TakeAction(s);
            //{ state, reward, terminated, truncated };
            auto [s1, r, b, t] = m_CartPoleEnv.step(a);

            vList.push_back({ s, a, r, s1, b });

            rewardCount += r;
            done = b;
            s = s1;

        }

        Update(actorAdam, criticAdam, vList);

        if (i % 10 == 0)
        {
            cout << "train i: " << i << " / " << maxCount << " , rewardCount: " << rewardCount << endl;
        }
    }



    TestData();
}

void ActorCritic::Update(torch::optim::Adam& actor, torch::optim::Adam& critic, QwList& vList)
{


    auto [s0, a, r, s1, done] = QwListToTensor(vList);

    auto v0 = m_CriticNet->forward(s0);
    auto v1 = r + m_dbGamma * m_CriticNet->forward(s1) * (1- done);
    
    auto td = v1 - v0; // Ê±Ðò²î·ÖÎó²î

    auto action = m_ActorNet->forward(s0).gather(1, a);
    auto logProbs = torch::log(action + 1e-8);
    auto actorLoss = torch::mean(-logProbs * td.detach());

    auto criticLoss = torch::mean(torch::mse_loss(v0, v1.detach()));

    actor.zero_grad();
    critic.zero_grad();

    actorLoss.backward();
    criticLoss.backward();
   

    actor.step();
    critic.step();

}


void ActorCritic::TestData()
{
    m_ActorNet->eval();
    m_CriticNet->eval();

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


