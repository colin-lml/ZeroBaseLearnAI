#include "pch.h"
#include "DDPG.h"

double DDPG::TakeAction(VectorDouble& s0,bool bPredict)
{
    torch::NoGradGuard noGrad;

    auto s = VectorDoubleTensor(s0, m_device);
    auto actionTensor = m_actor->forward(s);

    auto action = actionTensor.squeeze().item<double>();

    if (!bPredict)
    {
        auto noise = torch::randn({ m_objEnv->GetActionDim()}, m_device) * m_dbSigma;
        //cout << noise<<endl;
        action = action + noise.squeeze().item<double>();
    }

    action = std::clamp(action,m_objEnv->GetActionLow(),m_objEnv->GetActionHigh());

    return action;
}

void DDPG::GenerateTrainData(int maxCount)
{
    cout << "Currently DDPG" << endl;

    m_stateDim = m_objEnv->GetStateDim();
    m_actionDim = m_objEnv->GetActionDim();
	auto actionBound = m_objEnv->GetActionHigh();
    m_actor = PolicyNetCont(m_stateDim, m_actionDim, actionBound);
    m_targetActor = PolicyNetCont(m_stateDim, m_actionDim, actionBound);

    m_critic = QValueNetCont(m_stateDim, m_actionDim);
    m_targetCritic = QValueNetCont(m_stateDim, m_actionDim);

    m_actor->to(m_device);
    m_targetActor->to(m_device);

    m_critic->to(m_device);
    m_targetCritic->to(m_device);

    CopyModuleParameters(*m_actor, *m_targetActor);
    CopyModuleParameters(*m_critic, *m_targetCritic);

    m_actorOptimizer = std::make_unique<torch::optim::Adam>(m_actor->parameters(),torch::optim::AdamOptions(m_actorLearningRate));
    m_criticOptimizer = std::make_unique<torch::optim::Adam>(m_critic->parameters(),torch::optim::AdamOptions(m_criticLearningRate));

    m_actor->train();
    m_critic->train();

    m_targetActor->eval();
    m_targetCritic->eval();

    BaseAdvanced::GenerateTrainData(maxCount);

    m_actor->eval();
    m_critic->eval();

    m_actorOptimizer.reset();
    m_criticOptimizer.reset();
}

void DDPG::TrainGenerateItem1(const QwItem& item)
{
    AddCartPoleDataList(item);
}
void DDPG::TrainGenerateItem2(const QwList& vList)
{

}

void DDPG::SoftUpdate(torch::nn::Module& source,torch::nn::Module& target)
{
    torch::NoGradGuard noGrad;

    auto sourceParameters = source.parameters();
    auto targetParameters = target.parameters();

    TORCH_CHECK(sourceParameters.size() == targetParameters.size(),"DDPG::SoftUpdate: parameter count mismatch");

    for (size_t i = 0; i < sourceParameters.size(); ++i)
    {
        targetParameters[i].mul_(1.0 - m_tau);
        targetParameters[i].add_(sourceParameters[i],m_tau);
    }
}

