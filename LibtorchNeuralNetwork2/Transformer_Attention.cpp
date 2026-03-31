#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>
using namespace std;


class SelfAttention : public torch::nn::Module
{
public:
	SelfAttention(int64_t dim)
	{
		auto linear = torch::nn::LinearOptions(dim, dim).bias(false);

		Q = register_module("q", torch::nn::Linear(linear));
		K = register_module("k", torch::nn::Linear(linear));
		V = register_module("v", torch::nn::Linear(linear));
		
		norm_fact = 1.0 / sqrt(dim);
		
		std::vector<float> weight_vec(dim * dim, 1.0);
		auto weight = torch::tensor(weight_vec, torch::kFloat).view({ dim ,dim });
	
		Q->weight.set_data(weight);
		K->weight.set_data(weight);
		V->weight.set_data(weight);

	}
	/// x: [batch,seq, dim]
	/// x: [seq, batch, dim]
	auto forward(torch::Tensor x)
	{
		auto q = Q->forward(x);
		auto k = K->forward(x);
		auto v = V->forward(x);
		auto kt= k.transpose(0, 1);
		std::cout << std::fixed << std::setprecision(4);
		
		cout << "q k v \n" << q << endl;
		///cout << "k \n" << k << endl;
		///cout << "v \n" << v << endl;
		cout << "kt \n" << kt << endl;

		auto attn_score = torch::matmul(q, kt);
		cout << "q X kt \n" << attn_score << endl;

		attn_score = attn_score * norm_fact;
		cout << "scale q.X.kt  \n" << attn_score << endl;

		attn_score = torch::softmax(attn_score, -1);
		cout << "torch::softmax q.X.kt  \n" << attn_score << endl;

		auto out= torch::matmul(attn_score,v);

		cout << "torch::matmul V  \n" << out << endl;

		return out;
	}

	torch::nn::Linear Q{ nullptr };
	torch::nn::Linear K{ nullptr };
	torch::nn::Linear V{ nullptr };
	double norm_fact = 0 ;
};




void TransformerAttentionMain()
{
	auto x =  torch::tensor({
						    {{1.0, 0.0, 0.0, 0.0},
						     {2.0, 0.0, 0.0, 0.0},
						     {3.0, 0.0, 0.0, 0.0},
							 {4.0, 0.0, 0.0, 0.0},
							 {0.0, 0.0, 0.0, 0.0},
							 {0.0, 0.0, 0.0, 0.0}
							}},torch::kFloat);

	x.squeeze_();

	cout <<"input\n" << x << endl;

	auto atten= SelfAttention(x.size(1));
	auto y = atten.forward(x);
	cout <<"SelfAttention \n" << y << endl;

}