#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>

#include "TransformerTestData.h"

using namespace std;

#include "json.hpp"
using json = nlohmann::json;


typedef std::vector<int64_t>  WordListOnly;

TableVocab total_vocab = {
	{"Pad",PadId},
	{"S",1},
	{"E",2}
};

int64_t total_vocab_size = total_vocab.size();

WordListOnly GetWordIdOnly(TableVocab& vocabId, std::string data)
{
	std::vector<int64_t> input;
	for (const auto ch : Split(data))
	{
		if (vocabId.find(ch) == vocabId.end())
		{
			int64_t index = vocabId.size();
			vocabId.emplace(ch, index);

		}
		input.push_back(vocabId.at(ch));
	}
	return input;
}



class translatDatasetOnly : public torch::data::Dataset<translatDatasetOnly>
{
public:

	translatDatasetOnly()
	{
		wordCount.push_back(GetWordIdOnly(total_vocab, "Welcome 欢 迎 Pad Pad Pad Pad Pad Pad"));
		wordCount.push_back(GetWordIdOnly(total_vocab, "Welcome to PyTorch Tutorials  欢 迎 来 到 派 托 奇 教 程"));
		wordCount.push_back(GetWordIdOnly(total_vocab, "Welcome to PyTorch Tutorials  欢 迎 来 到 派 托 奇 教 程"));
		wordCount.push_back(GetWordIdOnly(total_vocab, "Welcome to Machine Learning  欢 迎 来 到 机 器 学 习 Pad"));
		//wordCount.push_back(GetWordIdOnly(total_vocab, "Welcome to PyTorch Tutorials"));
		//wordCount.push_back(GetWordIdOnly(total_vocab, "Welcome to Machine Learning"));
		total_vocab_size = total_vocab.size();
	}

	torch::optional<size_t> size() const
	{
		return wordCount.size();
	}

	torch::data::Example<torch::Tensor, torch::Tensor>  get(size_t index) override
	{
		auto item = wordCount[index];

		auto input = torch::tensor(item, torch::kLong);
		auto target = torch::tensor(item, torch::kLong);


		return { input, target };
	}

public:
	std::vector<WordListOnly> wordCount;

};



class DecoderLayerOnlyImpl : public torch::nn::Module
{
public:
	DecoderLayerOnlyImpl(int64_t dim, int64_t head, int64_t dff)
	{
		torch::nn::LayerNormOptions normOpt({ dim });
		norm1 = register_module("norm1", torch::nn::LayerNorm(normOpt));
		
		norm3 = register_module("norm3", torch::nn::LayerNorm(normOpt));
		ffn = register_module("ffn", FeedForwardNet(dim, dff));
		attention = register_module("attention", MultiHeadAttention(dim, head));	
	}

	auto forward(torch::Tensor& tgt, torch::Tensor mask)
	{

		auto y = attention->forward(tgt, tgt, tgt, mask);
		y = norm1->forward(tgt + y); //  残差连接

		auto y4 = ffn->forward(y);

		return norm3->forward(y + y4); //  残差连接
	}

private:

	torch::Tensor MaskAttention(torch::Tensor x, torch::Tensor mask)
	{
		auto y = attention->forward(x, x, x, mask);
		y = norm1->forward(x + y); //  残差连接
		return y;
	}

public:
	FeedForwardNet ffn{ nullptr };
	torch::nn::LayerNorm norm1{ nullptr }, norm3{ nullptr };
	MultiHeadAttention attention{ nullptr };

};

TORCH_MODULE(DecoderLayerOnly);


class DecodersOnlyImpl : public torch::nn::Module
{
public:
	DecodersOnlyImpl(int64_t dim, int64_t head, int64_t ffn, int64_t layers)
	{
		total_vocab_size = total_vocab.size();
		m_dim = dim;

		tgt_emb_ = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(total_vocab_size, dim)));
		pos_encoder = register_module("pos_encoder", PositionalEncoding(dim, total_vocab_size*10));
		fc = register_module("fc", torch::nn::Linear(dim, total_vocab_size));

		moduleLayers = register_module("moduleLayers2", torch::nn::ModuleList());

		for (int i = 0; i < layers; i++)
		{
			torch::nn::TransformerEncoderLayerOptions opt(dim, head);
			opt.dim_feedforward(ffn);
			opt.dropout(0);
			auto options = torch::nn::TransformerEncoderLayer(opt); 
			
			moduleLayers->push_back(options);
		}
	}

	auto forward(torch::Tensor& tgt)
	{
		
		auto tgt_mask = generate_square_subsequent_mask(tgt.size(1));

		//std::cout << "tgt_mask\n" << tgt_mask.sizes() << std::endl;
		//[batch, seq]  --> [seq, batch]
		
		tgt = tgt.permute({ 1,0 });


		tgt = tgt_emb_->forward(tgt) * std::sqrt(m_dim);
		tgt = pos_encoder->forward(tgt);


		for each(auto& item in * moduleLayers)
		{
			tgt = item->as<torch::nn::TransformerEncoderLayer>()->forward(tgt, tgt_mask);
		}

		return fc->forward(tgt);
		
	}
	torch::Tensor predict(string ch)
	{
		ch = "S " + ch;
		auto tgtpad = GetWordIdOnly(total_vocab, ch);
		

		int i = 0;
		while (i < total_vocab_size*2)
		{
			torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong);
			
			auto out = forward(tgt.unsqueeze(0));
			
			out = out.squeeze(-2);
			auto next_token = out.argmax(-1);
			int64_t key = next_token[i].item<int64_t>();
			tgtpad.push_back(key);
			//tgtpad.insert(tgtpad.begin(), );
			if ("E" == GetWordById(total_vocab, key))
			{
				break;
			}
			i++;
		}

		return torch::tensor(tgtpad, torch::kLong);
	}



	int64_t m_dim = 0;

	torch::nn::ModuleList moduleLayers{ nullptr };

	torch::nn::Embedding tgt_emb_{ nullptr };
	PositionalEncoding pos_encoder{ nullptr };

	torch::nn::Linear fc{ nullptr };

};

TORCH_MODULE(DecodersOnly);



void TrainData3(DecodersOnly& model, translatDatasetOnly& dataTrain)
{
	double accuracy = 0.03;
	auto datasetTrain = dataTrain.map(torch::data::transforms::Stack<>());
	auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(1));
	auto options = torch::nn::CrossEntropyLossOptions().ignore_index(PadId);
	torch::nn::CrossEntropyLoss loss_fn(options);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
	model->train();
	std::cout << "训练模型" << std::endl;

	for (int i = 0; i < max_train/2; i++)
	{
		float total_loss = 0;
		for (auto& item : *train_data_loader)
		{
			/// item.data, item.target : [batch, seq]

			auto [tgtInput, tgtOutput] = CreateDecoderInputOutput(item.target);
			//cout << "tgtInput\n" << tgtInput.sizes() << endl;
			//cout << "tgtOutput\n" << tgtOutput.sizes() << endl;
			
			auto tgtOut = model->forward(tgtInput);

			auto output = tgtOut.reshape({ -1, total_vocab_size });
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

void TestData3(DecodersOnly& model)
{

	model->eval();
	std::cout << "测试&翻译:" << std::endl;
	std::vector<std::string> tests;

	tests.push_back("Welcome to PyTorch Tutorials");
	tests.push_back("Welcome to Machine Learning");


	for (auto ch : tests)
	{
		auto result = model->predict(ch);

		// std::cout << std::regex_replace(ch, std::regex("Pad"), "") << " :  ";
		std::cout << ch << " :  ";


		for (int k = 0; k < result.numel(); k++)
		{
			std::cout << GetWordById(total_vocab, result[k].item<int64_t>()) << " ";
		}

		std::cout << std::endl;
	}

}



void DecoderOnlyMain()
{
	auto datasetTrain = translatDatasetOnly();
	DecodersOnly model(256, 8, 1024, 3);

	TrainData3(model, datasetTrain);

	TestData3(model);
}

