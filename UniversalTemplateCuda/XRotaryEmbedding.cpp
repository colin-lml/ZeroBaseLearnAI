#include "pch.h"
#include "XRotaryEmbedding.h"

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


XRotaryEmbeddingImpl::XRotaryEmbeddingImpl()
{

}

torch::Tensor XRotaryEmbeddingImpl::forward(const torch::Tensor& x)
{
   auto freqsCis  =  PrecomputeFreqs(x);
   return  ApplyRotary(x, freqsCis);
}

torch::Tensor XRotaryEmbeddingImpl::PrecomputeFreqs(const torch::Tensor& x)
{
    auto S = x.size(0);
    auto D = x.size(1);
    auto q = torch::arange(0, D, 2) / D;

    auto f = 1.0 / torch::pow(base, q);
 
    auto pos = torch::arange(S);
   
    if (S > maxSeqLen)
    {
        auto factor = S / maxSeqLen;  
        auto beta_fast = 16;  //
        auto beta_slow = 1;  // 

        auto [low, high] = find_correction_range(beta_fast, beta_slow, D, base, maxSeqLen);
        auto   smooth = 1 - linear_ramp_factor(low, high, D / 2);
        f = f / factor * (1 - smooth) + f * smooth;
    }

    auto freqs = torch::outer(pos, f);
    auto freqs_cis = torch::polar(torch::ones_like(freqs), freqs);

    return freqs_cis;
}

torch::Tensor XRotaryEmbeddingImpl::ApplyRotary(torch::Tensor x, const torch::Tensor& freqsCis)
{
    const auto orig_dtype = x.scalar_type();

    x = x.to(torch::kFloat32);
    auto S = x.size(0);
    auto D = x.size(1);
    x = x.view({ S,D / 2,2 });

    // 3. torch.view_as_complex
    x = torch::view_as_complex(x); //[S,D/2]

    torch::Tensor xMul = x * freqsCis;

    torch::Tensor y = torch::view_as_real(xMul);
    
    y = y.flatten(1);

    return y.to(orig_dtype);
}