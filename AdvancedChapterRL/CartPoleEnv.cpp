#include "pch.h"
#include "CartPoleEnv.h"

QwItemTensor QwListToTensor(const QwList& item, const torch::DeviceType& device)
{
 
    int64_t B = item.size();
    VectorDouble vS0;
    VectorDouble vS1;
    VectorDouble vA;
    VectorDouble vD;
    VectorDouble vR;
    int M = get<0>(item[0]).size();
    vS0.reserve(B* M);
    vS1.reserve(B * M);
    vA.reserve(B);
    vD.reserve(B);
    vR.reserve(B);

    for (auto& i :item)
    {
        auto [s0,a,r,s1, done] = i;  // s0, a, r, s1, done
        vS0.insert(vS0.end(), s0.begin(), s0.end());
        vS1.insert(vS1.end(), s1.begin(), s1.end());
        vA.push_back(a);
        vD.push_back(done);
        vR.push_back(r);
    }

    auto S0 = torch::tensor(vS0, torch::kFloat32).reshape({ B, M}).to(device);
    auto S1 = torch::tensor(vS1, torch::kFloat32).reshape({ B, M }).to(device);
    auto R = torch::tensor(vR, torch::kFloat32).reshape({ B, 1 }).to(device);
    auto A = torch::tensor(vA, torch::kFloat32).reshape({ B, 1 }).to(device);
    auto E = torch::tensor(vD, torch::kFloat32).reshape({ B, 1 }).to(device);


    return { S0,A,R,S1, E};

}

torch::Tensor VectorDoubleTensor(const VectorDouble& item, const torch::DeviceType& device)
{
    auto S0 =  torch::tensor(item, torch::kFloat32).reshape({ 1,-1 }).to(device);
    return S0;
}


XRandom::XRandom(int64_t x)
{
    std::random_device rd;
    if (x < 0)
    {
        m_gen.seed(rd());
    }
    else
    {
        m_gen.seed(x);
    }

}

int XRandom::RandInt(int min, int max)
{
    std::uniform_int_distribution<int> rand(min, max);
    return rand(m_gen);
}

double XRandom::RandDouble(double min, double max)
{
    std::uniform_real_distribution<double> rand(min, max);
    return rand(m_gen);
}



CartPoleEnv::CartPoleEnv()
{
    m_vResetState.resize(OBS_DIM, 0.0);
}

VectorDouble CartPoleEnv::reset()
{
    
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    for (int i = 0; i < OBS_DIM; ++i)
    {
        m_vResetState[i] = m_random.RandDouble(-0.05, 0.05);
    }
    return m_vResetState;
}

RewardState CartPoleEnv::step(double action)
{
    double x = m_vResetState[0];
    double x_dot = m_vResetState[1];
    double theta = m_vResetState[2];
    double theta_dot = m_vResetState[3];

 
    double force = (((int)(action)) == 1) ? force_mag : -force_mag;

    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);

   
    double temp = (force + mass_pole * pole_half_len * theta_dot * theta_dot * sin_t) / total_mass;
    double theta_acc = (gravity * sin_t - cos_t * temp)
        / (pole_half_len * (4.0 / 3.0 - mass_pole * cos_t * cos_t / total_mass));
    double x_acc = temp - mass_pole * pole_half_len * theta_acc * cos_t / total_mass;

  
    x_dot += dt * x_acc;
    x += dt * x_dot;
    theta_dot += dt * theta_acc;
    theta += dt * theta_dot;

    m_vResetState = { x, x_dot, theta, theta_dot };

   
    bool terminated = (std::fabs(x) > x_threshold) || (std::fabs(theta) > theta_threshold);
    bool truncated = false;
    double reward = terminated ? 0.0 : 1.0;

    return { m_vResetState, reward, terminated, truncated };
}


