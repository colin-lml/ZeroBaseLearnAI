#include <torch/torch.h>
#include <iostream>
#include <torch/serialize.h>
#include <regex>
//#include <iostream>
#include <fstream>
#include "TransformerTestData.h"
#include "Tokenizer.h"

using namespace std;
#define  maxtrain    1000*5

static	int64_t  gBOS = 0;
static	int64_t  gEOS = 0;
static	int64_t  gPad = 0;
static	int64_t	gCorpusVocabCount = 0;

class translatDatasetOnly : public torch::data::Dataset<translatDatasetOnly>
{
public:

	translatDatasetOnly()
	{
		m_dataToken.InitLoadDataSrc();
		m_vdata = m_dataToken.GetEncodeData();
		gCorpusVocabCount = m_dataToken.GetCorpusVocabCount();

		gPad = GetPad();
		gBOS = GetBOS();
		gEOS = GetEOS();

		for (auto& item : m_vdata)
		{
			m_nMaxTitle = max(m_nMaxTitle, item.title.size());
			m_nMaxAuthor = max(m_nMaxAuthor, item.author.size());
			m_nMaxContent = max(m_nMaxContent, item.content.size());
		}

		int count = 0;
		for (auto& item : m_vdata)
		{
			VectorCodeID input;
			input.push_back(gBOS);
			auto addPadLen = m_nMaxContent - item.content.size();
			input.insert(input.end(), item.content.begin(), item.content.end());
			for (size_t i = 0; i < addPadLen; i++)
			{
				input.push_back(gPad);
			}

			InputContent.push_back(input);

			input.clear();
			input.insert(input.end(), item.content.begin(), item.content.end());
			input.push_back(gEOS);
			for (size_t i = 0; i < addPadLen; i++)
			{
				input.push_back(gPad);
			}

			LableContent.push_back(input);

			if (2 <= ++count)
			{
				break;
			}	
		}


	}

	torch::optional<size_t> size() const
	{
		return InputContent.size();
	}

	torch::data::Example<torch::Tensor, torch::Tensor>  get(size_t index) override
	{
		
		auto& in = InputContent.at(index);
		
		auto inpput = torch::tensor(in, torch::kLong);


		auto& out = LableContent.at(index);
		
		auto lable = torch::tensor(out, torch::kLong);
		
		return {inpput, lable};

	}
	std::vector<int64_t> GetTangshiCode(std::string& line)
	{
		return m_dataToken.Encode(line);
	}
	std::string GetTangshiString(std::vector<int64_t>& vList)
	{
		return m_dataToken.Decode(vList);
	}
	INT64 GetPad()
	{
		return m_dataToken.GetPAD();
	}

	INT64 GetEOS()
	{
		return m_dataToken.GetEOS();
	}
	INT64 GetBOS()
	{
		return m_dataToken.GetBOS();
	}

public:
	
	std::vector<VectorCodeTangshi> m_vdata;
	Tokenizer m_dataToken;
	size_t m_nMaxTitle = 0;
	size_t m_nMaxAuthor = 0;
	size_t m_nMaxContent = 0;

	std::vector<VectorCodeID> InputContent;
	std::vector <VectorCodeID> LableContent;
};

#if 0

class DecodersOnlyImpl : public torch::nn::Module
{
public:
	DecodersOnlyImpl(int64_t dim, int64_t head, int64_t ffn, int64_t layers)
	{
		
		m_dim = dim;

		tgt_emb_ = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(gCorpusVocabCount, dim)));
		//pos_encoder = register_module("pos_encoder", torch::nn::Embedding(torch::nn::EmbeddingOptions(gCorpusVocabCount, dim)));
		pos_encoder = register_module("pos_encoder", PositionalEncoding(dim, max_vocab_len));

		fc = register_module("fc", torch::nn::Linear(dim, gCorpusVocabCount));

		moduleLayers = register_module("moduleLayers2", torch::nn::ModuleList());

		for (int i = 0; i < layers; i++)
		{
			torch::nn::TransformerEncoderLayerOptions opt(dim, head);
			opt.dim_feedforward(ffn);
			//opt.dropout(0);
			auto options = torch::nn::TransformerEncoderLayer(opt); 
			
			moduleLayers->push_back(options);
		}
	}

	auto forward(torch::Tensor& tgt)
	{
		
		int64_t seq = tgt.size(1);
		int64_t batch = tgt.size(0);
		auto tgt_mask = generate_square_subsequent_mask(seq);
		auto tgt_key_padding_mask = (tgt == gPad).to(torch::kBool);  // [batch,seq]
		//std::cout << "tgt_mask\n" << tgt_mask << std::endl;
		//std::cout << "tgt_key_padding_mask\n" << tgt_key_padding_mask << std::endl;
		//[batch, seq]  --> [seq, batch]
		
		//torch::Tensor pos = torch::arange(0, seq);
		//pos = pos.unsqueeze(0).repeat({ batch, 1 });

		//tgt = tgt.permute({ 1,0 });
		tgt = tgt_emb_->forward(tgt);
		tgt = pos_encoder->forward(tgt);

		///tgt = tgt + pos2;

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
		ch = "<BOS>" + ch;
		
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
			if (key == dataTest.GetEOS())
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
	//torch::nn::Embedding pos_encoder{ nullptr };
	PositionalEncoding pos_encoder{ nullptr };

	torch::nn::Linear fc{ nullptr };

};

TORCH_MODULE(DecodersOnly);



void TrainData3(DecodersOnly& model, translatDatasetOnly& dataTrain)
{
	double accuracy = 0.05;
	auto datasetTrain = dataTrain.map(torch::data::transforms::Stack<>());
	auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(300));
	auto options = torch::nn::CrossEntropyLossOptions().ignore_index(dataTrain.GetPad());
	torch::nn::CrossEntropyLoss loss_fn(options);
	//torch::nn::functional::cross_entropy loss_fn(options);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
	model->train();
	std::cout << "ÑµÁ·Ä£ÐÍ" << std::endl;

	auto start_time = chrono::high_resolution_clock::now();

	
	for (int i = 0; i < maxtrain; i++)
	{
		uint64_t k = 0;
		float total_loss = 0;
		for (auto& item : *train_data_loader)
		{
			/// item.data, item.target : [batch, seq]
			auto tgtInput = item.data;
			auto tgtOutput = item.target;

			///BBPE  kkk;
			
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
			///std::cout <<i<< " ... "<<k++ << std::endl;
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
	std::cout << "²âÊÔ:" << std::endl;
	std::vector<std::string> tests;

	tests.push_back("»­ÌÃ´º");
	tests.push_back("¾ÆÈª×Ó");
	
	//tests.push_back("ÐÛ¹Ø×èÈû´÷Áé÷¡");
	//tests.push_back("·üÓê³¯º®Ï¤²»Ê¤");
	//tests.push_back("ÖÚÄñ¸ß·É¾¡");
	//tests.push_back("ËÉÏÂÎÊÍ¯×Ó");


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
	DecodersOnly model(512, 8, 1024, 1);

	std::string model_path = "Decoder_Only_model3.pt";
	std::ifstream filem(model_path);
	bool bmodel = filem.is_open();
	if (!bmodel)
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

#else

vector<vector<int64_t>>  Tensor2dToVector(torch::Tensor tensor)
{
	tensor = tensor.cpu().contiguous().to(torch::kLong);
	

	int d0 = tensor.size(0);  
	int d1 = tensor.size(1);  
	

	vector<vector<int64_t>> res(d0, vector<int64_t>(d1));

	int64_t* data = tensor.data_ptr<int64_t>();

	for (int i = 0; i < d0; i++)
	{
		std::copy(data + i * d1, data + (i + 1) * d1, res[i].begin());
	}
		
	return res;
}

class EmbeddingWithPositionImpl : public torch::nn::Module
{
public:
	EmbeddingWithPositionImpl(int64_t d_model, int64_t vocab_size, int64_t max_len)
	{
		m_iDmodel = d_model;
		m_iMaxlen = max_len;
		m_posEncode = torch::zeros({ m_iMaxlen, m_iDmodel }, torch::kFloat32);

		Encoding();
		m_emb = register_module("tgt_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, m_iDmodel)));
		register_buffer("posEncode", m_posEncode);
	}


	torch::Tensor forward(torch::Tensor x)
	{
		// x [bath, seq]
		x = m_emb->forward(x);

		// x [bath, seq, dim]

		auto dim = x.size(1);
		//cout << "x  " << x.sizes() << endl;
		//cout << "m_posEncode  "  << m_posEncode.sizes() << endl;
		//cout << "m_posEncode.slice  " << m_posEncode.slice(1, 0, dim).sizes() << endl;
		x = x + m_posEncode.slice(1, 0, dim);

		return  x;
	}

private:
	void Encoding()
	{
		auto pos = torch::arange(0, m_iMaxlen, torch::kFloat32).reshape({ m_iMaxlen, 1 });
		auto den_indices = torch::arange(0, m_iDmodel, 2, torch::kFloat32);
		auto den = torch::exp(-den_indices * std::log(10000.0f) / m_iDmodel);
		m_posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, m_iDmodel, 2) }, torch::sin(pos * den));
		m_posEncode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, m_iDmodel, 2) }, torch::cos(pos * den));
		m_posEncode.unsqueeze_(0);
		// [bath,seq,dim]

	}

	torch::nn::Embedding m_emb{ nullptr };
	torch::Tensor m_posEncode;
	int64_t m_iDmodel = 0;
	int64_t m_iMaxlen = 0;
};

TORCH_MODULE(EmbeddingWithPosition);

struct DeOnlyOptions
{
	int64_t dmodel = 512;
	int64_t head = 8;
	int64_t ffn = 2048;
	int64_t layers = 6;
	int64_t vocab_size = 300;
	int64_t max_len = 500;
};


class DecodersOnlyImpl : public torch::nn::Module
{

public:

	DecodersOnlyImpl(const DeOnlyOptions& inOpt)
	{
		fc = register_module("fc", torch::nn::Linear(inOpt.dmodel, inOpt.vocab_size));

		m_emb = register_module("m_emb", EmbeddingWithPosition(inOpt.dmodel, inOpt.vocab_size, inOpt.max_len));

		moduleLayers = register_module("EncoderLayers", torch::nn::ModuleList());

		m_option = inOpt;

		for (int i = 0; i < inOpt.layers; i++)
		{
			torch::nn::TransformerEncoderLayerOptions opt(inOpt.dmodel, inOpt.head);
			opt.dim_feedforward(inOpt.ffn);
			opt.dropout(0);

			moduleLayers->push_back(torch::nn::TransformerEncoderLayer(opt));
		}

		//torch::nn::TransformerEncoderLayer()
	}

	auto forward(torch::Tensor x)
	{
		// cout <<x.sizes()<< endl;
		// x [bath, seq]
		int64_t batch = x.size(0);
		int64_t seq = x.size(1);
		auto src_mask = generate_square_subsequent_mask(seq);

		auto tgt_key_padding_mask = (x == gPad).to(torch::kBool);

		x = m_emb->forward(x);

		x = x.permute({1, 0, 2 });
		//cout <<x.sizes()<< endl;

		for (auto& item : *moduleLayers)
		{
			x = item->as<torch::nn::TransformerEncoderLayer>()->forward(x, src_mask, tgt_key_padding_mask);

		}

		return fc->forward(x);
	}

	string predict(string ch, translatDatasetOnly& dataTest)
	{
		ch = "<BOS>" + ch;
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
			if (key == gEOS)
			{
				break;
			}
			i++;
		}

		return dataTest.GetTangshiString(tgtpad);

	}


	torch::Tensor generate_square_subsequent_mask(int64_t sz)
	{
		auto mask = torch::triu(torch::ones({ sz, sz }, torch::kFloat32), 1);

		mask = mask.masked_fill(mask == 1, -std::numeric_limits<float>::infinity());
		return mask;
	}


public:
	EmbeddingWithPosition m_emb{ nullptr };
	torch::nn::ModuleList moduleLayers{ nullptr };
	torch::nn::Linear fc{ nullptr };
	DeOnlyOptions m_option;
};
TORCH_MODULE(DecodersOnly);

#define max_train  1000*5
#define batchsize   4

void TrainData3(DecodersOnly& model, translatDatasetOnly& dataTrain)
{
	double accuracy = 0.15;
	auto datasetTrain = dataTrain.map(torch::data::transforms::Stack<>());
	auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(batchsize));
	auto options = torch::nn::CrossEntropyLossOptions().ignore_index(gPad);
	torch::nn::CrossEntropyLoss loss_fn(options);
	//torch::nn::functional::cross_entropy loss_fn(options);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(std::min(std::sqrt(batchsize) * 1e-4, 0.85)));
	model->train();
	std::cout << "ÑµÁ·Ä£ÐÍ" << std::endl;

	auto start_time = chrono::high_resolution_clock::now();


	for (int i = 0; i < maxtrain; i++)
	{
		float total_loss = 0;
		for (auto& item : *train_data_loader)
		{
			auto out = model->forward(item.data);
			out = out.reshape({ -1, out.size(2) });
			optimizer.zero_grad();

			auto tgt = item.target.reshape({ -1 });

			auto loss = loss_fn(out, tgt);

			total_loss += loss.item<float>();
			torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
			loss.backward();
			optimizer.step();
			/* 
			if ( 0)
			{
				auto input = Tensor2dToVector(item.data);
				auto input2 = Tensor2dToVector(item.target);
				for (size_t i = 0; i < input.size(); i++)
				{
					auto str = dataTrain.GetTangshiString(input[i]);
					auto str2 = dataTrain.GetTangshiString(input2[i]);
					cout << str << endl;
					cout << str2 << endl << endl;
				}

				cout << endl << endl;
				cout << endl << endl;
			}
			*/					
		}

		if (i % 1 == 0 || (i + 1 == maxtrain))
		{
			std::cout << "i: " << i + 1 << " / " << maxtrain << " , loss: " << total_loss << std::endl;
		}

		if (total_loss <= accuracy)
		{
			std::cout << "i: " << i + 1 << " / " << maxtrain << " , loss: " << total_loss << std::endl << std::endl;
			break;
		}
	}
}

void TestData3(DecodersOnly& model, translatDatasetOnly& dataTest)
{
	model->eval();
	std::cout << "²âÊÔ:" << std::endl;
	std::vector<std::string> tests;

	tests.push_back("É½Ò»³Ì");
	tests.push_back("ÈËÉúÈôÖ»Èç³õ¼û");

	for (auto ch : tests)
	{
		auto result = model->predict(ch, dataTest);

		// std::cout << std::regex_replace(ch, std::regex("Pad"), "") << " :  ";

		std::cout << ch << " :" << std::endl;
		std::cout << result << std::endl;;

		std::cout << std::endl;
	}

}

void DecoderOnlyMain()
{
	auto datasetTrain = translatDatasetOnly();

	DeOnlyOptions opt;
	opt.dmodel = 512;
	opt.head = 8;
	opt.ffn = 2048;
	opt.layers = 1;
	opt.max_len = 800;
	opt.vocab_size = gCorpusVocabCount;
	
	DecodersOnly model(opt);

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

#endif