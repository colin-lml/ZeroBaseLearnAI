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
	SelfAttention()
	{


	}
	void InitQKV(int64_t dim)
	{
		auto linear = torch::nn::LinearOptions(dim, dim).bias(false);

		Q = register_module("q", torch::nn::Linear(linear));
		K = register_module("k", torch::nn::Linear(linear));
		V = register_module("v", torch::nn::Linear(linear));

		norm_fact = 1.0 / sqrt(dim); // 缩放

		auto onesw = torch::eye(dim); //单位矩阵

		Q->weight.set_data(onesw);
		K->weight.set_data(onesw);
		V->weight.set_data(onesw);
	}

	
	auto forward(torch::Tensor x,torch::Tensor mask = {})
	{
		torch::Tensor q ;
		torch::Tensor k ;
		torch::Tensor v;
		torch::Tensor kt;
		torch::Tensor out;

		auto dim = x.dim();
	

		if (dim == 3)
		{
			//x:  [batch, seq, dim]  --->  [seq, batch, dim]
			x = x.permute({1,0,2});
			InitQKV(x.size(2));
		}
		else
		{
			// x: [seq, dim]
			InitQKV(x.size(1));
		}
		/// 1.输入x 与 q k v 运算  q k v是 单位矩阵所以 q k v = x
		 q = Q->forward(x);
		 k = K->forward(x);
		 v = V->forward(x);

		 cout << "q k v \n" << q << endl;

		if (dim == 3)
		{
			kt = k.permute({ 1,2,0 });
		 	v = v.permute({ 1,0,2 });
		}
		else
		{
			kt = k.transpose(0, 1);// kt 是 k 的置换矩阵  kt: [dim,seq]
		}

	
		cout << "kt \n" << kt << endl;

		auto attn_score = torch::matmul(q, kt); //2.  
		cout << "q X kt \n" << attn_score << endl;

		attn_score = attn_score * norm_fact;     //3. 矩阵缩放
		cout << "scale q.X.kt  \n" << attn_score << endl;

		if (mask.defined())
		{
			attn_score += mask;
		}


		attn_score = torch::softmax(attn_score, -1);//4. Softmax 归一化指数函数
		cout << "torch::softmax q.X.kt  \n" << attn_score << endl;

		out = torch::matmul(attn_score, v); /// 5. qKt * V

		cout << "torch::matmul V  \n" << out << endl;
	
		return out;
	}

	torch::nn::Linear Q{ nullptr };
	torch::nn::Linear K{ nullptr };
	torch::nn::Linear V{ nullptr };
	double norm_fact = 0 ;
};

class MultiHeadAttention: public torch::nn::Module
{
public:
	void InitQKV(int64_t dim, int64_t head=2)
	{
		assert(dim % head == 0);

		auto linear = torch::nn::LinearOptions(dim, dim).bias(false);

		Q = register_module("q", torch::nn::Linear(linear));
		K = register_module("k", torch::nn::Linear(linear));
		V = register_module("v", torch::nn::Linear(linear));
		Wo = register_module("Wo", torch::nn::Linear(linear)); // 输出投影

		norm_fact = 1.0 / sqrt(dim);
		
		Dk = dim / head;
		H = head;

		auto onesw = torch::eye(dim);   
		Q->weight.set_data(onesw);
		K->weight.set_data(onesw);
		V->weight.set_data(onesw);
		Wo->weight.set_data(onesw);
	}

	auto forward(torch::Tensor x, int64_t head = 2, torch::Tensor mask = {})
	{
		x.squeeze_(); //x: [batch, seq ,dim]  -->   [seq, dim]
		assert(x.dim() == 2);
		//x: [seq, dim]
		  
		InitQKV(x.size(1), head);

		auto seq = x.size(0);
		auto dim = x.size(1);

		auto q = Q->forward(x);
		auto k = K->forward(x);
		auto v = V->forward(x);
		q = q.view({ seq,H,Dk }); //q: [seq, dim] ->   [S, H, Dk] 
		k = k.view({ seq,H,Dk });
		v = v.view({ seq,H,Dk });

		q = q.permute({ 1,0,2 }); //[S, H, Dk] --->[H, S, Dk]
		k = k.permute({ 1,0,2 });
		v = v.permute({ 1,0,2 });

		cout << "q k v \n" << q << endl;

		auto kt = k.permute({ 0,2,1 }); //kt:  [H, S, Dk] --> [H, Dk, S]

		cout << "kt \n" << kt << endl;

		auto attn_score = torch::matmul(q, kt);  // [H, S, Dk] *  [H, Dk, S]
		cout << "q X kt \n" << attn_score << endl;


		attn_score = attn_score * norm_fact;
		cout << "scale q.X.kt  \n" << attn_score << endl;

		if (mask.defined())
		{
			attn_score += mask;
		}

		attn_score = torch::softmax(attn_score, -1); /// attn_score: [H, S, S]
		cout << "torch::softmax q.X.kt  \n" << attn_score.squeeze() << endl;

		auto out = torch::matmul(attn_score, v); // [H, S, S] * [H, S, Dk]  ->  out: [H, S, Dk]
		out = out.transpose(1, 0).contiguous().view({ seq, dim }); //  [H, S, Dk] --> [S, H, Dk] -> [seq, dim]

		cout << "torch::matmul QK * V  \n" << out.squeeze() << endl;
		
		out = Wo->forward(out);

		return out;
	}


	auto forward2(torch::Tensor x, int64_t head = 2,torch::Tensor mask = {})
	{
		assert(x.dim() == 3);
	  
		x = x.permute({ 1,0,2 });  //   x: x: [batch, seq, dim]-->  [seq, batch, dim]

		InitQKV(x.size(2), head);

		auto seq = x.size(0);
		auto batch = x.size(1);
		auto dim = x.size(2);

		auto q = Q->forward(x);
		auto k = K->forward(x);
		auto v = V->forward(x);
		q = q.view({ seq,batch,H,Dk}); //q: [seq, batch, dim] ->   [S, B, H, Dk] 
		k = k.view({ seq,batch,H,Dk });
		v = v.view({ seq,batch,H,Dk });
		
		q = q.permute({1,2,0,3}); //[S, B, H, Dk] --->[B, H, S, Dk]
		k = k.permute({ 1,2,0,3 });
		v = v.permute({ 1,2,0,3 });
		
		cout << "q k v \n" << q << endl;

		auto kt = k.permute({ 0,1,3,2}); //kt:  [B, H, S, Dk] --> [B, H, Dk, S]

		cout << "kt \n" << kt.squeeze() << endl;

		auto attn_score = torch::matmul(q, kt);
		cout << "q X kt \n" << attn_score << endl;


		attn_score = attn_score * norm_fact;
		cout << "scale q.X.kt  \n" << attn_score << endl;

		if (mask.defined())
		{
			attn_score += mask;
		}

		attn_score = torch::softmax(attn_score, -1); /// attn_score: [B, H, S, S]
		cout << "torch::softmax q.X.kt  \n" << attn_score << endl;

		auto out = torch::matmul(attn_score, v); // [B, H, S, S] * [B, H, S, Dk]  ->  out: [B, H, S, Dk]
		out = out.transpose(1, 2).contiguous().view({ seq,batch, dim }); //  [B, H, S, Dk] --> [B, S, H, Dk] -> [seq,batch, dim]
		
		cout << "torch::matmul QK * V  \n" << out << endl;
		
		out = Wo->forward(out);

		return out;
	}


	torch::nn::Linear Q{ nullptr };
	torch::nn::Linear K{ nullptr };
	torch::nn::Linear V{ nullptr };
	torch::nn::Linear Wo{ nullptr };

	double norm_fact = 0;
	int64_t Dk;
	int64_t H;
};


void TransformerAttentionMain()
{

	auto x = torch::tensor({
				{{1.0, 0.0, 0.0, 0.0}, // Welcome
				 {2.0, 0.0, 0.0, 0.0}, // to
				 {3.0, 0.0, 0.0, 0.0}, // Machine
				 {4.0, 0.0, 0.0, 0.0}, // Learning
				 {0.0, 0.0, 0.0, 0.0}, // Pad
				 {0.0, 0.0, 0.0, 0.0}  // Pad
				} }, torch::kFloat);


	cout << "input\n" << x << endl;
	
	cout << "-------------SelfAttention--------------------\n"  << endl;
	auto x1 = x.squeeze();
	auto atten= SelfAttention();
	auto y = atten.forward(x1);
	cout << "-------------SelfAttention--------------------\n"  << endl;
	

	cout << "\n\n-------------MultiHeadAttention--------------------\n"  << endl;
	auto multiAtten = MultiHeadAttention();
	multiAtten.forward2(x,1);
	cout << "-------------MultiHeadAttention--------------------\n"  << endl;
}


/*
input
(1,.,.) =
  1  0  0  0
  2  0  0  0
  3  0  0  0
  4  0  0  0
  0  0  0  0
  0  0  0  0
[ CPUFloatType{1,6,4} ]
-------------SelfAttention--------------------

q k v
 1  0  0  0
 2  0  0  0
 3  0  0  0
 4  0  0  0
 0  0  0  0
 0  0  0  0
[ CPUFloatType{6,4} ]
kt
 1  2  3  4  0  0
 0  0  0  0  0  0
 0  0  0  0  0  0
 0  0  0  0  0  0
[ CPUFloatType{4,6} ]
q X kt
  1   2   3   4   0   0
  2   4   6   8   0   0
  3   6   9  12   0   0
  4   8  12  16   0   0
  0   0   0   0   0   0
  0   0   0   0   0   0
[ CPUFloatType{6,6} ]
scale q.X.kt
 0.5000  1.0000  1.5000  2.0000  0.0000  0.0000
 1.0000  2.0000  3.0000  4.0000  0.0000  0.0000
 1.5000  3.0000  4.5000  6.0000  0.0000  0.0000
 2.0000  4.0000  6.0000  8.0000  0.0000  0.0000
 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
[ CPUFloatType{6,6} ]
torch::softmax q.X.kt
 0.0904  0.1490  0.2457  0.4052  0.0548  0.0548
 0.0313  0.0851  0.2314  0.6291  0.0115  0.0115
 0.0086  0.0386  0.1731  0.7758  0.0019  0.0019
 0.0021  0.0158  0.1170  0.8645  0.0003  0.0003
 0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
 0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
[ CPUFloatType{6,6} ]
torch::matmul V
 2.7463  0.0000  0.0000  0.0000
 3.4122  0.0000  0.0000  0.0000
 3.7084  0.0000  0.0000  0.0000
 3.8426  0.0000  0.0000  0.0000
 1.6667  0.0000  0.0000  0.0000
 1.6667  0.0000  0.0000  0.0000
[ CPUFloatType{6,4} ]
-------------SelfAttention--------------------



nput
(1,.,.) =
  1  0  0  0
  2  0  0  0
  3  0  0  0
  4  0  0  0
  0  0  0  0
  0  0  0  0
[ CPUFloatType{1,6,4} ]


-------------MultiHeadAttention--------------------

q k v
(1,.,.) =
  1  0  0  0
  2  0  0  0
  3  0  0  0
  4  0  0  0
  0  0  0  0
  0  0  0  0
[ CPUFloatType{1,6,4} ]
kt
(1,.,.) =
  1  2  3  4  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
[ CPUFloatType{1,4,6} ]
q X kt
(1,.,.) =
   1   2   3   4   0   0
   2   4   6   8   0   0
   3   6   9  12   0   0
   4   8  12  16   0   0
   0   0   0   0   0   0
   0   0   0   0   0   0
[ CPUFloatType{1,6,6} ]
scale q.X.kt
(1,.,.) =
  0.5000  1.0000  1.5000  2.0000  0.0000  0.0000
  1.0000  2.0000  3.0000  4.0000  0.0000  0.0000
  1.5000  3.0000  4.5000  6.0000  0.0000  0.0000
  2.0000  4.0000  6.0000  8.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
[ CPUFloatType{1,6,6} ]
torch::softmax q.X.kt
 0.0904  0.1490  0.2457  0.4052  0.0548  0.0548
 0.0313  0.0851  0.2314  0.6291  0.0115  0.0115
 0.0086  0.0386  0.1731  0.7758  0.0019  0.0019
 0.0021  0.0158  0.1170  0.8645  0.0003  0.0003
 0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
 0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
[ CPUFloatType{6,6} ]
torch::matmul QK * V
 2.7463  0.0000  0.0000  0.0000
 3.4122  0.0000  0.0000  0.0000
 3.7084  0.0000  0.0000  0.0000
 3.8426  0.0000  0.0000  0.0000
 1.6667  0.0000  0.0000  0.0000
 1.6667  0.0000  0.0000  0.0000
[ CPUFloatType{6,4} ]
-------------MultiHeadAttention--------------------
*/