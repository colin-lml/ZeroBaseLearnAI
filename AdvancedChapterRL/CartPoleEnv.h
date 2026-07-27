#pragma once

using  RewardState = std::tuple<std::vector<double>, double, bool, bool>;
using  VectorDouble = std::vector<double>;

using  QwItem = std::tuple<VectorDouble, int, double, VectorDouble, bool>;
using  QwList = vector<QwItem>;
using  QwList2D = vector<QwList>;
using  QwItemTensor = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

QwItemTensor QwListToTensor(const QwList& item, const torch::DeviceType& device);
torch::Tensor VectorDoubleTensor(const VectorDouble& item, const torch::DeviceType& device);


class XRandom
{
public:
    XRandom(int64_t x = -1);
    int RandInt(int min, int max);
    double RandDouble(double min, double max);
    std::mt19937& GetGen()
    {
        return m_gen;
    }
private:

	std::mt19937 m_gen;
};


class CartPoleEnv
{
public:

    static constexpr int OBS_DIM = 4;
    static constexpr int ACT_DIM = 2;
 
    const double gravity = 9.8;
    const double mass_cart = 1.0;
    const double mass_pole = 0.1;
    const double total_mass = mass_cart + mass_pole;
    const double pole_half_len = 0.5;
    const double force_mag = 10.0;
    const double dt = 0.02;

   
    const double x_threshold = 2.4;
    const double theta_threshold = 12.0 * M_PI / 180.0;

    //  [x, x_dot, theta, theta_dot]
    VectorDouble m_vResetState;
    XRandom m_random;

    CartPoleEnv();

    int GetStateDim()
    {
        return OBS_DIM;
    }

    int GetActionDim()
    {
        return ACT_DIM;
    }

    VectorDouble reset();

    RewardState step(int action);

};

