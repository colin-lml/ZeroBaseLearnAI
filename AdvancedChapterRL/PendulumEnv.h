#pragma once


class PendulumEnv : public Env
{
public:
    static constexpr int OBS_DIM = 3;
    static constexpr int ACT_DIM = 1;

    static constexpr double gravity = 10.0;
    static constexpr double mass = 1.0;
    static constexpr double length = 1.0;
    static constexpr double max_speed = 8.0;
    static constexpr double max_torque = 2.0;
    static constexpr double dt = 0.05;
    static constexpr int max_episode_steps = 200;

    static constexpr double pi = 3.14159265358979323846;

    // 状态：[cos(theta), sin(theta), theta_dot]
    VectorDouble m_state;
    XRandom m_random;
    int m_stepCount = 0;

    PendulumEnv() = default;

    int GetStateDim() override
    {
        return OBS_DIM;
    }

    int GetActionDim() override
    {
        return ACT_DIM;
    }

    double GetActionLow() override
    {
        return -max_torque;
    }

    double GetActionHigh() override
    {
        return max_torque;
    }

    VectorDouble reset() override;

    RewardState step(double action) override;
};