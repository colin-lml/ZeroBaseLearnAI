#pragma once

class CartPoleEnv
{
public:

    static constexpr int OBS_DIM = 4;
    static constexpr int ACT_DIM = 2;
    static constexpr int MAX_EP_STEPS = 500;
    static constexpr double M_PI2 = 3.14159265358979323846;
    // 物理参数（与Gymnasium CartPole-v1 完全对齐）
    const double gravity = 9.8;
    const double mass_cart = 1.0;
    const double mass_pole = 0.1;
    const double total_mass = mass_cart + mass_pole;
    const double pole_half_len = 0.5;
    const double force_mag = 10.0;
    const double dt = 0.02;

    // 终止阈值
    const double x_threshold = 2.4;
    const double theta_threshold = 12.0 * M_PI2 / 180.0;

    // 状态 [x, x_dot, theta, theta_dot]
    std::vector<double> state;
    std::mt19937 rng;

    CartPoleEnv();

    // 重置环境，返回初始观测
    std::vector<double> reset();

    // 执行一步动作
    // 返回：(观测, 奖励, terminated, truncated)
    std::tuple<std::vector<double>, double, bool, bool> step(int action);

    // 打印观测信息
    void print_obs(const std::vector<double>& obs) const;

};

