#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>

#include "TransformerTestData.h"

using namespace std;



class EncoderLayerImpl : public torch::nn::Module
{
public:
	EncoderLayerImpl(int64_t dim, int64_t head, int64_t dff)
	{
		torch::nn::LayerNormOptions normOpt({ dim });
		norm1 = register_module("norm1", torch::nn::LayerNorm(normOpt));
		norm2 = register_module("norm2", torch::nn::LayerNorm(normOpt));
		ffn = register_module("ffn", FeedForwardNet(dim, dff));
		attention = register_module("attention", MultiHeadAttention(dim, head));
	}

	auto forward(torch::Tensor x)
	{
		auto y = attention->forward(x,x,x);

		y = norm1->forward(x + y); ///  残差连接

		auto y2 = ffn->forward(y);

		return norm2->forward(y + y2); ///  残差连接
	}

	FeedForwardNet ffn{ nullptr };
	torch::nn::LayerNorm norm1{ nullptr }, norm2{ nullptr };
	MultiHeadAttention attention{ nullptr };
};

TORCH_MODULE(EncoderLayer);

class EncodersImpl : public torch::nn::Module
{
public:
	EncodersImpl(int64_t dim, int64_t head, int64_t ffn, int64_t layers)
	{
		moduleLayers = register_module("moduleLayers", torch::nn::ModuleList());

		for (int i = 0; i < layers; i++)
		{
			moduleLayers->push_back(EncoderLayer(dim, head, ffn));
		}
	}

	auto forward(torch::Tensor x)
	{
	
		for each(auto& item in *moduleLayers)
		{
			x = item->as<EncoderLayer>()->forward(x);
		}
		return x;
	}
	torch::nn::ModuleList moduleLayers{ nullptr };
	
};

TORCH_MODULE(Encoders);




class DecoderLayerImpl : public torch::nn::Module
{
public:
	DecoderLayerImpl(int64_t dim, int64_t head, int64_t dff)
	{
		torch::nn::LayerNormOptions normOpt({ dim });
		norm1 = register_module("norm1", torch::nn::LayerNorm(normOpt));
		norm2 = register_module("norm2", torch::nn::LayerNorm(normOpt));
		norm3 = register_module("norm3", torch::nn::LayerNorm(normOpt));
		ffn = register_module("ffn", FeedForwardNet(dim, dff));
		attention = register_module("attention", MultiHeadAttention(dim, head));
		attention2 = register_module("attention2", MultiHeadAttention(dim, head));
	}

	auto forward(torch::Tensor& tgt, torch::Tensor& memory,torch::Tensor tgtmask)
	{
		
		auto y = MaskAttention(tgt, tgtmask);

		//cout << "y\n" << y.sizes() << endl;
		//cout << "memory\n" << memory.sizes() << endl;

		auto y2 = attention2->forward(y, memory, memory);

		auto y3 = norm2->forward(y+y2); //  残差连接

		auto y4 = ffn->forward(y3);

		return norm3->forward(y3 + y4); //  残差连接
	}

private:

	torch::Tensor MaskAttention(torch::Tensor x, torch::Tensor mask)
	{
		auto y = attention->forward(x,x,x, mask);
		y = norm1->forward(x + y); //  残差连接
		return y;
	}

public:
	FeedForwardNet ffn{ nullptr };
	torch::nn::LayerNorm norm1{ nullptr }, norm2{ nullptr }, norm3{ nullptr };
	MultiHeadAttention attention{ nullptr };
	MultiHeadAttention attention2{ nullptr };
};

TORCH_MODULE(DecoderLayer);


class DecodersImpl : public torch::nn::Module
{
public:
	DecodersImpl(int64_t dim, int64_t head, int64_t ffn, int64_t layers)
	{
		moduleLayers = register_module("moduleLayers2", torch::nn::ModuleList());

		for (int i = 0; i < layers; i++)
		{
			moduleLayers->push_back(DecoderLayer(dim, head, ffn));
		}
	}

	auto forward(torch::Tensor& tgt, torch::Tensor& memory, torch::Tensor tgtmask)
	{

		for each(auto& item in * moduleLayers)
		{
			tgt = item->as<DecoderLayer>()->forward(tgt, memory, tgtmask);
		}

		return tgt;
	}

	torch::nn::ModuleList moduleLayers{ nullptr };

};

TORCH_MODULE(Decoders);


class MyTransformerImpl : public torch::nn::Module
{
public:
	MyTransformerImpl(int64_t dim, int64_t head, int64_t ffn, int64_t layerEncoder, int64_t layerDecoder)
	{
		src_emb = register_module("src_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(src_vocab_size, dim)));
		tgt_emb_ = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(tgt_vocab_size, dim)));
		pos_encoder = register_module("pos_encoder", PositionalEncoding(dim, max_vocab_len));
		encoders = register_module("Encoders", Encoders(dim, head, ffn, layerEncoder));
		decoders = register_module("Decoders", Decoders(dim, head, ffn, layerDecoder));
		fc = register_module("fc", torch::nn::Linear(dim, tgt_vocab_size));
	}

	torch::Tensor forward(torch::Tensor src, torch::Tensor tgt)
	{
		auto none_mask = torch::Tensor();

		auto tgt_mask = generate_square_subsequent_mask(tgt.size(1));
	

		//[batch, seq]  --> [seq, batch]
		src = src.permute({ 1,0 });
		tgt = tgt.permute({ 1,0 });

		//std::cout << "input " << src << std::endl;
		src = src_emb->forward(src) * std::sqrt(dim_model);
		src = pos_encoder->forward(src);

		tgt = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
		tgt = pos_encoder->forward(tgt);
	

		return TransformerForward(src, tgt, tgt_mask);
	}

	torch::Tensor predict(torch::Tensor src)
	{
		//std::cout << src << std::endl;

		auto srcemb = src_emb->forward(src) * std::sqrt(dim_model);
		srcemb = pos_encoder->forward(srcemb);
		auto memory = encoders->forward(srcemb);

		std::vector<int64_t> tgtpad = GetWordId(tgt_vocab, "S");

		int i = 0;
		while (i < tgt_vocab_size * 2)
		{
			torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong);
			auto tgt_mask = generate_square_subsequent_mask(tgt.size(0));
			///std::cout << "tgt_mask " << tgt_mask << std::endl;
			auto tgt_emb = tgt_emb_->forward(tgt) * std::sqrt(dim_model);
			tgt_emb = pos_encoder->forward(tgt_emb);
			auto out = decoders->forward(tgt_emb, memory, tgt_mask);
			out = fc->forward(out).squeeze(-2);
			auto next_token = out.argmax(-1);
			int64_t key = next_token[i].item<int64_t>();
			tgtpad.push_back(key);
			//tgtpad.insert(tgtpad.begin(), );
			if ("E" == GetWordById(tgt_vocab, key))
			{
				break;
			}
			i++;
		}

		return torch::tensor(tgtpad, torch::kLong);
	}

private:
	torch::Tensor TransformerForward(torch::Tensor& src,torch::Tensor& tgt,torch::Tensor tgtmask)
	{
		 auto outputEncoder = encoders->forward(src);
		 auto  outputDecoder = decoders->forward(tgt, outputEncoder, tgtmask);
		 return fc->forward(outputDecoder);
	}

	Encoders encoders{nullptr};
	Decoders decoders{nullptr};

	torch::nn::Embedding src_emb{ nullptr };
	torch::nn::Embedding tgt_emb_{ nullptr };
	PositionalEncoding pos_encoder{ nullptr };

	torch::nn::Linear fc{ nullptr };
};

TORCH_MODULE(MyTransformer);

void TestData2(MyTransformer& model)
{
	 
	model->eval();
	std::cout << "测试&翻译:" << std::endl;
	std::vector<std::string> tests;
	tests.push_back("Welcome");
	tests.push_back("Welcome to");
	tests.push_back("Welcome to PyTorch");
	tests.push_back("Welcome to Machine");
	tests.push_back("Welcome to PyTorch Tutorials");
	tests.push_back("Welcome to Machine Learning");

	tests.push_back("Learning");
	tests.push_back("Tutorials");
	tests.push_back("PyTorch Tutorials");
	tests.push_back("Machine Learning");

	for (auto ch : tests)
	{
		auto item = GetWordId(src_vocab, ch);
		auto src = torch::tensor(item, torch::kLong);

		auto result = model->predict(src);

		// std::cout << std::regex_replace(ch, std::regex("Pad"), "") << " :  ";
		std::cout << ch << " :  ";


		for (int k = 0; k < result.numel(); k++)
		{
			std::cout << GetWordById(tgt_vocab, result[k].item<int64_t>()) << " ";
		}

		std::cout << std::endl;
	}
	
}




void TrainData2(MyTransformer& model)
{
	double accuracy = 0.03;

	auto datasetTrain = translatDataset().map(torch::data::transforms::Stack<>());
	auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(1));
	auto options = torch::nn::CrossEntropyLossOptions().ignore_index(PadId);
	torch::nn::CrossEntropyLoss loss_fn(options);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
	model->train();
	std::cout << "训练模型" << std::endl;

	for (int i = 0; i < max_train; i++)
	{
		float total_loss = 0;
		for (auto& item : *train_data_loader)
		{
			/// item.data, item.target : [batch, seq]

			auto [tgtInput, tgtOutput] = CreateDecoderInputOutput(item.target);

			auto tgtOut = model->forward(item.data, tgtInput);

			auto output = tgtOut.reshape({ -1, tgt_vocab_size });
			optimizer.zero_grad();
			auto tgt = tgtOutput.squeeze(0);

			auto loss = loss_fn(output, tgt);
			total_loss += loss.item<float>();
			torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
			loss.backward();

			optimizer.step();

		}


		if (i % 10 == 0 || (i + 1 == max_train))
		{
			std::cout << "i: " << i + 1 << " , loss: " << total_loss << std::endl;
		}

		if (total_loss <= accuracy)
		{
			std::cout << "i: " << i + 1 << " , loss: " << total_loss << std::endl << std::endl;
			break;
		}
	}
	std::cout << std::endl;
}




void HandwrittenTransformerMain()
{
	torch::manual_seed(6);
	std::string model_path = "MyTransformer_model2.pt";
	MyTransformer model(dim_model,2, dim_feed,1,1);

	std::ifstream filem(model_path);
	bool bmodel = filem.is_open();
	if (!bmodel || true)
	{
		TrainData2(model);
		torch::save(model, model_path);
	}
	else
	{
		torch::load(model, model_path);
		std::cout << "load model ...." << std::endl;
	}

	filem.close();

	TestData2(model);
}