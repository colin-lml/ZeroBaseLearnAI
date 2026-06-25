#include "pch.h"
#include "CartPoleEnv.h"






CartPoleEnv::CartPoleEnv()
{
    rng.seed(std::random_device{}());
    state.resize(OBS_DIM, 0.0);
}

std::vector<double> CartPoleEnv::reset()
{
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    for (int i = 0; i < OBS_DIM; ++i)
    {
        state[i] = dist(rng);
    }
    return state;
}

std::tuple<std::vector<double>, double, bool, bool> CartPoleEnv::step(int action)
{
    double x = state[0];
    double x_dot = state[1];
    double theta = state[2];
    double theta_dot = state[3];

    // 施加推力：0左，1右
    double force = (action == 1) ? force_mag : -force_mag;

    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);

    // 动力学核心公式
    double temp = (force + mass_pole * pole_half_len * theta_dot * theta_dot * sin_t) / total_mass;
    double theta_acc = (gravity * sin_t - cos_t * temp)
        / (pole_half_len * (4.0 / 3.0 - mass_pole * cos_t * cos_t / total_mass));
    double x_acc = temp - mass_pole * pole_half_len * theta_acc * cos_t / total_mass;

    // 欧拉积分更新状态
    x_dot += dt * x_acc;
    x += dt * x_dot;
    theta_dot += dt * theta_acc;
    theta += dt * theta_dot;

    state = { x, x_dot, theta, theta_dot };

    // 判断终止
    bool terminated = (std::fabs(x) > x_threshold) || (std::fabs(theta) > theta_threshold);
    bool truncated = false;
    double reward = terminated ? 0.0 : 1.0;

    return { state, reward, terminated, truncated };
}

void CartPoleEnv::print_obs(const std::vector<double>& obs) const
{
    std::cout << "x: " << obs[0]
        << " | x_dot: " << obs[1]
        << " | theta: " << obs[2]
        << " | theta_dot: " << obs[3] << "\n";
}
