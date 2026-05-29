// UniversalTemplate.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"


double  find_correction_dim(double num_rotations, double dim, double base, double max_seq_len)
{
    return dim * std::log(max_seq_len / (num_rotations * 2 * M_PI)) / (2 * std::log(base));
}
std::pair<double, double> find_correction_range(double low_rot, double high_rot, double dim, double base, double max_seq_len)
{
    double   low = std::floor(find_correction_dim(low_rot, dim, base, max_seq_len));
    double   high = std::ceil(find_correction_dim(high_rot, dim, base, max_seq_len));
    return { low,high };
}

torch::Tensor linear_ramp_factor(double min, double max, double dim)
{
    if (min == max)
    {
        max += 0.001;
    }
    
    auto    linear_func = (torch::arange(dim, torch::kFloat32) - min) / (max - min);
    auto    ramp_func = torch::clamp(linear_func, 0, 1);
    return ramp_func;
}


torch::Tensor PrecomputeFreqs(const torch::Tensor& x, double  base=10000, double max_seq_len=6)
{
    auto S = x.size(0);
    auto D = x.size(1);
    auto q = torch::arange(0, D, 2 )/ D;
  
    auto f = 1.0 / torch::pow(base, q);
    cout << f << endl;

    auto pos = torch::arange(S);
    cout << pos << endl;

    if (S > max_seq_len)
    {
        auto factor = 10.0;
        auto beta_fast = 16;
        auto beta_slow = 1;
        auto [low, high] = find_correction_range(beta_fast, beta_slow, D, base, max_seq_len);
        auto   smooth = 1 - linear_ramp_factor(low, high, D / 2);
        f = f / factor * (1 - smooth) + f * smooth;
    }

    auto freqs = torch::outer(pos, f);
    auto freqs_cis = torch::polar(torch::ones_like(freqs), freqs);

    return freqs_cis;
}

torch::Tensor ApplyRotaryEmb(torch::Tensor x, const torch::Tensor& freqs_cis) 
{
    
    const auto orig_dtype = x.scalar_type();

  
    x = x.to(torch::kFloat32);
    auto S = x.size(0);
    auto D = x.size(1);
    x = x.view({S,D/2,2}); 

    // 3. torch.view_as_complex
    x = torch::view_as_complex(x); //[S,D/2]

    torch::Tensor x_mul = x * freqs_cis;

    torch::Tensor y = torch::view_as_real(x_mul);
    cout <<"uuu\n" << y.flatten(1) << endl;;
    y = y.flatten(1);

    return y.to(orig_dtype);
}



int main()
{
    torch::manual_seed(12);

    //XTrainPredict xTrainPredict;
   /// xTrainPredict.TestData();

    //torch::Tensor t = torch::ones({ 6, 4 }, torch::kFloat32);
    //cout << t << endl;
    //auto freqs_cis = PrecomputeFreqs(t);
    //cout << "freqs_cis\n" << freqs_cis << endl;
    //auto rotary = ApplyRotaryEmb(t, freqs_cis);
    //cout <<"rotary\n" << rotary << endl;

    auto t2 = torch::ones({ 9, 4 }, torch::kFloat32);
    auto freqs_cis = PrecomputeFreqs(t2);
    auto rotary = ApplyRotaryEmb(t2, freqs_cis);
    cout <<"t2-rotary\n" << rotary << endl;

    std::cin.get();
}

