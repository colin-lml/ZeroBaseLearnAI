// ReinforcementLearning.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"

#if 0

static std::random_device rd;
static std::mt19937 gen(rd());
static constexpr double M_PI2 = 3.14159265358979323846;
class CartPoleEnv2
{
public:
    // 构造函数
    CartPoleEnv2(bool sutton_barto_reward = false)
        : sutton_barto_reward_(sutton_barto_reward),
        gravity_(9.8f),
        masscart_(1.0f),
        masspole_(0.1f),
        total_mass_(masscart_ + masspole_),
        length_(0.5f),  // 杆长的一半
        polemass_length_(masspole_* length_),
        force_mag_(10.0f),
        tau_(0.02f),    // 状态更新时间步
        kinematics_integrator_("euler"),
        theta_threshold_radians_(12 * 2 * M_PI2 / 360),  // ±12度
        x_threshold_(2.4f),
        steps_beyond_terminated_(-1) {

        // 初始化随机分布（-0.05 ~ 0.05）
        reset_dist_ = std::uniform_real_distribution<float>(-0.05f, 0.05f);

        // 初始化状态
        state_.resize(4, 0.0f);
    }

    // 重置环境
    std::vector<float> reset(int seed = -1, float low = -0.05f, float high = 0.05f)
    {
        // 设置随机种子
        if (seed >= 0)
        {
            gen.seed(seed);
        }

        // 重新初始化分布
        reset_dist_ = std::uniform_real_distribution<float>(low, high);

        // 随机初始化状态
        for (int i = 0; i < 4; ++i)
        {
            state_[i] = reset_dist_(gen);
        }

        steps_beyond_terminated_ = -1;
        return state_;
    }

    // 单步执行
    struct StepResult
    {
        std::vector<float> observation;
        float reward;
        bool terminated;
        bool truncated;
    };

    StepResult step(int action)
    {
        // 验证动作有效性
        if (action != 0 && action != 1)
        {
            throw std::invalid_argument("Invalid action: must be 0 or 1");
        }

        // 验证状态已初始化
        if (steps_beyond_terminated_ == -2)
        {
            throw std::runtime_error("Call reset before using step method");
        }

        // 解析状态
        float x = state_[0];
        float x_dot = state_[1];
        float theta = state_[2];
        float theta_dot = state_[3];

        // 计算作用力（0=左，1=右）
        float force = (action == 1) ? force_mag_ : -force_mag_;

        float costheta = cos(theta);
        float sintheta = sin(theta);

        // 物理计算（参考原论文）
        float temp = (force + polemass_length_ * theta_dot * theta_dot * sintheta) / total_mass_;
        float thetaacc = (gravity_ * sintheta - costheta * temp) /
            (length_ * (4.0f / 3.0f - masspole_ * costheta * costheta / total_mass_));
        float xacc = temp - polemass_length_ * thetaacc * costheta / total_mass_;

        // 积分更新状态
        if (kinematics_integrator_ == "euler")
        {
            x += tau_ * x_dot;
            x_dot += tau_ * xacc;
            theta += tau_ * theta_dot;
            theta_dot += tau_ * thetaacc;
        }
        else {  // semi-implicit euler
            x_dot += tau_ * xacc;
            x += tau_ * x_dot;
            theta_dot += tau_ * thetaacc;
            theta += tau_ * theta_dot;
        }

        // 更新状态
        state_[0] = x;
        state_[1] = x_dot;
        state_[2] = theta;
        state_[3] = theta_dot;

        // 判断终止条件
        bool terminated =
            (x < -x_threshold_) ||
            (x > x_threshold_) ||
            (theta < -theta_threshold_radians_) ||
            (theta > theta_threshold_radians_);

        // 截断条件（默认500步，可外部控制）
        bool truncated = false;

        // 计算奖励
        float reward = 0.0f;
        if (!terminated)
        {
            reward = sutton_barto_reward_ ? 0.0f : 1.0f;
        }
        else if (steps_beyond_terminated_ == -1)
        {
            // 首次终止
            steps_beyond_terminated_ = 0;
            reward = sutton_barto_reward_ ? -1.0f : 1.0f;
        }
        else
        {
            // 终止后继续调用step
            if (steps_beyond_terminated_ == 0)
            {
                std::cerr << "Warning: Calling step() after terminated=True. Call reset() first." << std::endl;
            }
            steps_beyond_terminated_++;
            reward = sutton_barto_reward_ ? -1.0f : 0.0f;
        }

        return { state_, reward, terminated, truncated };
    }

    // 获取当前状态
    std::vector<float> get_state() const {
        return state_;
    }

    // 获取动作空间大小（0/1）
    int get_action_space_size() const {
        return 2;
    }

    // 获取观测空间边界
    std::pair<std::vector<float>, std::vector<float>> get_observation_bounds() const
    {
        std::vector<float> low = {
            -x_threshold_ * 2,
            -INFINITY,
            -theta_threshold_radians_ * 2,
            -INFINITY
        };
        std::vector<float> high =
        {
            x_threshold_ * 2,
            INFINITY,
            theta_threshold_radians_ * 2,
            INFINITY
        };
        return { low, high };
    }

private:
    // 奖励模式
    bool sutton_barto_reward_;

    // 物理参数
    float gravity_;
    float masscart_;
    float masspole_;
    float total_mass_;
    float length_;
    float polemass_length_;
    float force_mag_;
    float tau_;
    std::string kinematics_integrator_;

    // 终止阈值
    float theta_threshold_radians_;
    float x_threshold_;

    // 状态
    std::vector<float> state_;
    int steps_beyond_terminated_;  // -1=未终止, >=0=终止后步数

    // 随机分布
    std::uniform_real_distribution<float> reset_dist_;
};

#endif




int main()
{
   
    /* 
    PolicyIteration policy;
    policy.Iteration();
    policy.ValueIteration2();

    FreeModel td;
    td.MonteCarloMethods();
    td.SarsaIteration();
    td.NStepSarsaIteration();
    td.QLearningIteration();
    td.DynaQIteration();
    */

    DeepQNetwork dqn;
    //dqn.PlayCartPole(200,true);

    DuelingDQN duelingDqn;
    //duelingDqn.PlayCartPole();
    
    Reinforce  p;
    p.PlayCartPole();

    cin.get();
}


