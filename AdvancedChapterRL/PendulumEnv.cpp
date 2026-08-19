#include "pch.h"
#include "PendulumEnv.h"

VectorDouble PendulumEnv::reset()
{
    const auto theta = m_random.RandDouble(-pi, pi);
    const auto thetaDot = m_random.RandDouble(-1.0, 1.0);

    m_state = {
        std::cos(theta),
        std::sin(theta),
        thetaDot
    };

    m_stepCount = 0;

    return m_state;
}

RewardState PendulumEnv::step(double action)
{
    if (m_state.size() != OBS_DIM)
    {
        reset();
    }

    const auto theta = std::atan2(m_state[1], m_state[0]);
    auto thetaDot = m_state[2];

    const auto torque = std::clamp(
        action,
        -max_torque,
        max_torque);

    // Pendulum-v1 的角度归一化到 [-pi, pi]。
    auto normalizedTheta = std::fmod(theta + pi, 2.0 * pi);
    if (normalizedTheta < 0.0)
    {
        normalizedTheta += 2.0 * pi;
    }
    normalizedTheta -= pi;

    // Pendulum-v1 动力学：
    //
    // theta_ddot =
    //     -3g / (2l) * sin(theta + pi)
    //     + 3 / (m*l^2) * torque
    const auto thetaAcceleration =
        -3.0 * gravity / (2.0 * length) *
        std::sin(normalizedTheta + pi)
        + 3.0 / (mass * length * length) * torque;

    thetaDot += thetaAcceleration * dt;
    thetaDot = std::clamp(
        thetaDot,
        -max_speed,
        max_speed);

    const auto nextTheta =
        normalizedTheta + thetaDot * dt;

    m_state = {
        std::cos(nextTheta),
        std::sin(nextTheta),
        thetaDot
    };

    // Pendulum-v1 reward：
    //
    // -(theta^2 + 0.1 * theta_dot^2 + 0.001 * u^2)
    const auto reward =
        -(
            normalizedTheta * normalizedTheta
            + 0.1 * thetaDot * thetaDot
            + 0.001 * torque * torque);

    ++m_stepCount;

    // Pendulum-v1 本身没有物理终止状态，
    // 但 TimeLimit wrapper 在 200 步后截断。
    const auto terminated = false;
    const auto truncated =
        m_stepCount >= max_episode_steps;

    return {
        m_state,
        reward,
        terminated,
        truncated
    };
}