#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>
#include "TransformerTestData.h"
#include "Tokenizer.h"

using namespace std;
#define  maxtrain    1000*10

static	int64_t	gCorpusVocabCount = 0;

class translatDatasetOnly : public torch::data::Dataset<translatDatasetOnly>
{
public:

	translatDatasetOnly()
	{
		m_dataToken.InitLoadDataSrc();
		m_vdata = m_dataToken.GetEncodeData();
		gCorpusVocabCount = m_dataToken.GetCorpusVocabCount();
	}

	torch::optional<size_t> size() const
	{
		return m_vdata.size();
	}

	torch::data::Example<>  get(size_t index) override
	{
		
		auto item = m_vdata.at(index);
		item.pop_back();
		auto inpput = torch::tensor(item, torch::kLong);

		item = m_vdata.at(index);
		item.erase(item.begin());
		auto lable = torch::tensor(item, torch::kLong);

		return {inpput, lable};
	}
	std::vector<int64_t> GetTangshiCode(std::string& line)
	{
		return m_dataToken.GetTangshiCode(line);
	}
	std::string GetTangshiString(std::vector<int64_t>& vList)
	{
		return m_dataToken.GetTangshiString(vList);
	}
public:
	
	std::vector<std::vector<int64_t>> m_vdata;
	Tokenizer m_dataToken;
};


class DecodersOnlyImpl : public torch::nn::Module
{
public:
	DecodersOnlyImpl(int64_t dim, int64_t head, int64_t ffn, int64_t layers)
	{
		
		m_dim = dim;

		tgt_emb_ = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(gCorpusVocabCount, dim)));
		pos_encoder = register_module("pos_encoder", torch::nn::Embedding(torch::nn::EmbeddingOptions(gCorpusVocabCount, dim)));
		fc = register_module("fc", torch::nn::Linear(dim, gCorpusVocabCount));

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
		
		int64_t seq = tgt.size(1);
		int64_t batch = tgt.size(0);
		auto tgt_mask = generate_square_subsequent_mask(seq);
		auto tgt_key_padding_mask = (tgt == PadId).to(torch::kBool);  // [batch,seq]
		//std::cout << "tgt_mask\n" << tgt_mask << std::endl;
		//std::cout << "tgt_key_padding_mask\n" << tgt_key_padding_mask << std::endl;
		//[batch, seq]  --> [seq, batch]
		
		torch::Tensor pos = torch::arange(0, seq);
		pos = pos.unsqueeze(0).repeat({ batch, 1 });

		tgt = tgt_emb_->forward(tgt) * std::sqrt(m_dim);
		torch::Tensor pos2  = pos_encoder->forward(pos);

		tgt = tgt + pos2;

		tgt = tgt.permute({ 1,0, 2});

		for each(auto& item in * moduleLayers)
		{
			tgt = item->as<torch::nn::TransformerEncoderLayer>()->forward(tgt, tgt_mask, tgt_key_padding_mask);
			//std::cout << "tgt\n" << tgt.sizes() << std::endl;
			//tgt = item->as<torch::nn::TransformerEncoderLayer>()->forward(tgt);
		}

		return fc->forward(tgt);
		
	}
	
	string predict(string ch, translatDatasetOnly& dataTest)
	{
		ch = "S" + ch;
		
		auto tgtpad = dataTest.GetTangshiCode(ch);
		

		int i = 0;
		while (i < 100)
		{
			torch::Tensor tgt = torch::tensor(tgtpad, torch::kLong);		
			auto out = forward(tgt.unsqueeze(0));

			out = out.squeeze(-2);
			auto next_token = out.argmax(-1);
			int64_t key = next_token[i].item<int64_t>();
			tgtpad.push_back(key);
			if (key == 2)
			{
				break;
			}
			i++;
		}

		return dataTest.GetTangshiString(tgtpad);
	}



	int64_t m_dim = 0;

	torch::nn::ModuleList moduleLayers{ nullptr };

	torch::nn::Embedding tgt_emb_{ nullptr };
	torch::nn::Embedding pos_encoder{ nullptr };

	torch::nn::Linear fc{ nullptr };

};

TORCH_MODULE(DecodersOnly);



void TrainData3(DecodersOnly& model, translatDatasetOnly& dataTrain)
{
	double accuracy = 0.05;
	auto datasetTrain = dataTrain.map(torch::data::transforms::Stack<>());
	auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(30));
	auto options = torch::nn::CrossEntropyLossOptions().ignore_index(PadId);
	torch::nn::CrossEntropyLoss loss_fn(options);
	//torch::nn::functional::cross_entropy loss_fn(options);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
	model->train();
	std::cout << "训练模型" << std::endl;

	auto start_time = chrono::high_resolution_clock::now();

	for (int i = 0; i < maxtrain; i++)
	{
		float total_loss = 0;
		for (auto& item : *train_data_loader)
		{
			/// item.data, item.target : [batch, seq]
			auto tgtInput = item.data;
			auto tgtOutput = item.target;
			
			//cout << "tgtInput\n" << tgtInput << endl;
			//cout << "tgtOutput\n" << tgtOutput<< endl;
			
			auto tgtOut = model->forward(tgtInput);

			///cout << "tgtOut\n" << tgtOut.sizes() << endl;
			
			auto output = tgtOut.reshape({ -1, tgtOut.size(2)});
			optimizer.zero_grad();
			auto tgt = tgtOutput.reshape({ -1 });
	
			auto loss = loss_fn(output, tgt);
			total_loss += loss.item<float>();
			torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
			loss.backward();

			optimizer.step();

		}


		if (i % 5 == 0 || (i + 1 == maxtrain))
		{
			std::cout << "i: " << i + 1 << " / " << maxtrain << " , loss: " << total_loss << std::endl;
		}

		if (total_loss <= accuracy)
		{
			std::cout << "i: " << i + 1 <<" / " << maxtrain <<" , loss: " << total_loss << std::endl << std::endl;
			break;
		}
	}
	std::cout << std::endl;

	auto end_time = chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
	std::cout << "end-time: " << duration_ms <<" seconds " << endl;
}

void TestData3(DecodersOnly& model, translatDatasetOnly& dataTest)
{

	model->eval();
	std::cout << "测试:" << std::endl;
	std::vector<std::string> tests;

	tests.push_back("静夜思");
	tests.push_back("相思");
	
	tests.push_back("床前明月光");
	tests.push_back("白日依山尽");
	//tests.push_back("众鸟高飞尽");
	//tests.push_back("松下问童子");


	for (auto ch : tests)
	{
		auto result = model->predict(ch, dataTest);

		// std::cout << std::regex_replace(ch, std::regex("Pad"), "") << " :  ";

		std::cout << ch << " :"<<std::endl;
		std::cout << result << std::endl;;
		
		std::cout << std::endl;
	}

}



void DecoderOnlyMain()
{
	
	auto datasetTrain = translatDatasetOnly();
	DecodersOnly model(128, 4, 512, 2);

	std::string model_path = "Decoder_Only_model3.pt";
	std::ifstream filem(model_path);
	bool bmodel = filem.is_open();
	if (!bmodel||true)
	{
		TrainData3(model, datasetTrain);
		torch::save(model, model_path);
	}
	else
	{
		torch::load(model, model_path);
		std::cout << "load model ...." << std::endl;
	}

	filem.close();

	TestData3(model, datasetTrain);
}

