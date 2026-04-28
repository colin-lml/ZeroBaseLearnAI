// LibtorchSimpleNeuralNetwork.cpp: 定义应用程序的入口点。
//


#include <stdlib.h> 
#include "torch/torch.h"
#include <chrono>
#include "Tokenizer.h"
#include <windows.h>

#include "BBPE.h"

using namespace std;
void CnnMain();
void RnnMain();
int autogradMain();
void EmbeddingMain();
void ResNetMain();
void TransformerMain();
void TransformerAttentionMain();
void HandwrittenTransformerMain();
void DecoderOnlyMain();

struct NetModule : torch::nn::Module
{
	NetModule()
	{
		fc1 = register_module("fc1", torch::nn::Linear(2, 2));
		fc2 = register_module("fc2", torch::nn::Linear(2, 2));
		fc1->to(torch::kDouble);
		fc2->to(torch::kDouble);


		fc1->weight.set_data(torch::tensor({ {0.15,0.20},
											 {0.25, 0.30} }, torch::kDouble));
		fc1->bias.set_data(torch::tensor({ 0.35,0.35 }, torch::kDouble));


		fc2->weight.set_data(torch::tensor({ {0.40,0.45}
										   ,{0.50,0.55} }, torch::kDouble));
		fc2->bias.set_data(torch::tensor({ 0.60,0.60 }, torch::kDouble));


		cout << "<<<-------------------------------------------------->>>" << endl;
		std::cout << std::fixed << std::setprecision(10);
		cout << fc1->weight << endl << fc1->bias << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << fc2->weight << endl << fc2->bias << endl;
		cout << "<<<<-------------------------------------------------- >>>" << endl << endl;

	}

	torch::Tensor forward(torch::Tensor x)
	{
		//cout << fc1->forward(x) << endl  << endl;
		auto t = torch::sigmoid(fc1->forward(x));
		auto l2 = torch::sigmoid(fc2->forward(t));

		return 	l2;
	}



	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

};








#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <regex>
#include <cstdint>
#include <cstring>

using namespace std;

// 自定义 vector<uint8_t> 的哈希函数（用于 unordered_map 做 key）
struct VectorHash 
{
    size_t operator()(const vector<uint8_t>& v) const 
    {
        size_t hash = 0;
        for (uint8_t b : v) 
        {
            hash ^= std::hash<uint8_t>{}(b)+0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

class BPETokenizer
{
private:
    map<vector<uint8_t>, uint32_t> b2i;       // bytes -> id（有序）
    unordered_map<uint32_t, vector<uint8_t>> i2b; // id -> bytes
    uint32_t next_id = 0;

    unordered_map<string, uint32_t> sp_s2i;   // special token str -> id
    unordered_map<uint32_t, string> sp_i2s;   // id -> special token str

    // 统计相邻 token 频率
    void pair_stats(const vector<vector<uint8_t>>& tokens, unordered_map<vector<uint8_t>, int, VectorHash>& stats)
    {
        for (size_t i = 0; i < tokens.size() - 1; ++i) 
        {
            vector<uint8_t> merged;
            int a = tokens[i].size();
            int b = tokens[i+1].size();

            merged.reserve(tokens[i].size() + tokens[i + 1].size());
            merged.insert(merged.end(), tokens[i].begin(), tokens[i].end());
            merged.insert(merged.end(), tokens[i + 1].begin(), tokens[i + 1].end());
            stats[merged]++;
        }
    }

    // 合并指定 token
    vector<vector<uint8_t>> merge_pair(const vector<vector<uint8_t>>& tokens,const vector<uint8_t>& target)
    {
        vector<vector<uint8_t>> res;
        size_t i = 0;
        while (i < tokens.size()) 
        {
            if (i + 1 < tokens.size()) 
            {
                vector<uint8_t> combined;
                combined.reserve(tokens[i].size() + tokens[i + 1].size());
                combined.insert(combined.end(), tokens[i].begin(), tokens[i].end());
                combined.insert(combined.end(), tokens[i + 1].begin(), tokens[i + 1].end());
                if (combined == target) 
                {
                    res.push_back(combined);
                    i += 2;
                    continue;
                }
            }
            res.push_back(tokens[i]);
            i++;
        }
        return res;
    }

public:
    BPETokenizer() 
    {
        // 初始化 0~255 基础字节
        for (int i = 0; i < 256; ++i) 
        {
            b2i[{(uint8_t)i}] = i;
        }
        next_id = 256;
        rebuild_i2b();
    }

    void rebuild_i2b() 
    {
        i2b.clear();
        for (auto& p : b2i)
        {
            i2b[p.second] = p.first;
        }
    }

    // 训练 BPE
    void train(const vector<string>& text_list, uint32_t vocab_size)
    {
        vector<vector<vector<uint8_t>>> tokens_list;

        for (auto& s : text_list) 
        {
            vector<vector<uint8_t>> tokens;
            for (uint8_t c : s)
            {
                tokens.push_back({ c });
            }
            tokens_list.push_back(tokens);
        }

        while (next_id < vocab_size) 
        {
            unordered_map<vector<uint8_t>, int, VectorHash> stats;
            for (auto& tokens : tokens_list) 
            {
                pair_stats(tokens, stats);
            }

            if (stats.empty())
            {
                break;
            }

            vector<uint8_t> best_pair;
            int max_cnt = 0;
            for (auto& p : stats) 
            {
                if (p.second > max_cnt) 
                {
                    max_cnt = p.second;
                    best_pair = p.first;
                }
            }

            for (auto& tokens : tokens_list)
            {
                tokens = merge_pair(tokens, best_pair);
            }
            //string str(best_pair.begin(), best_pair.end());
           // cout << next_id <<" ,  " << str << endl;
                 
            b2i[best_pair] = next_id++;
            
            rebuild_i2b();
        }

    }

    // 添加特殊 token
    void add_special_tokens(const vector<string>& tokens)
    {
        for (auto& s : tokens) 
        {
            if (sp_s2i.count(s)) continue;
            sp_s2i[s] = next_id;
            sp_i2s[next_id] = s;
            next_id++;
        }
    }

    // 编码：文本 → ids, tokens
    pair<vector<uint32_t>, vector<vector<uint8_t>>> encode(const string& text)
    {
        vector<uint32_t> ids;
        vector<vector<uint8_t>> tokens_out;

        if (!sp_s2i.empty())
        {
            string pat;
            for (auto& p : sp_s2i) 
            {
                if (!pat.empty()) pat += "|";
                pat += regex_replace(p.first, regex(R"([.^$|*+?()\[\]{}])"), R"(\$&)");
            }
            regex reg("(" + pat + ")");
            sregex_token_iterator it(text.begin(), text.end(), reg, { -1, 0 });
            sregex_token_iterator end;

            for (auto seg = it; seg != end; ++seg)
            {
                string s = *seg;
                if (s.empty()) continue;

                if (sp_s2i.count(s)) 
                {
                    uint32_t id = sp_s2i[s];
                    ids.push_back(id);
                    tokens_out.emplace_back(s.begin(), s.end());
                    continue;
                }

                vector<vector<uint8_t>> tokens;
                for (uint8_t c : s) tokens.push_back({ c });

                while (true) 
                {
                    unordered_map<vector<uint8_t>, int, VectorHash> stats;
                    pair_stats(tokens, stats);
                    if (stats.empty()) break;

                    vector<uint8_t> best;
                    uint32_t min_id = UINT32_MAX;
                    for (auto& p : stats)
                    {
                        auto& tok = p.first;
                        if (!b2i.count(tok)) continue;
                        uint32_t id = b2i[tok];
                        if (id < min_id) {
                            min_id = id;
                            best = tok;
                        }
                    }
                    if (best.empty()) break;
                    tokens = merge_pair(tokens, best);
                }

                for (auto& t : tokens) 
                {
                    ids.push_back(b2i[t]);
                    tokens_out.push_back(t);
                }
            }
        }
        return { ids, tokens_out };
    }

    // 解码
    string decode(const vector<uint32_t>& ids) 
    {
        vector<uint8_t> bytes;
        for (uint32_t id : ids)
        {
            if (sp_i2s.count(id))
            {
                auto& s = sp_i2s[id];
                bytes.insert(bytes.end(), s.begin(), s.end());
            }
            else if (i2b.count(id)) 
            {
                auto& b = i2b[id];
                bytes.insert(bytes.end(), b.begin(), b.end());
            }
        }
        return string(bytes.begin(), bytes.end());
    }

    // 保存
    void save(const string& path) 
    {
        ofstream f(path, ios::binary);
        uint32_t n = b2i.size();
        f.write((char*)&n, 4);
        for (auto& p : b2i) {
            uint32_t len = p.first.size();
            f.write((char*)&len, 4);
            f.write((char*)p.first.data(), len);
            f.write((char*)&p.second, 4);
        }

        uint32_t m = sp_s2i.size();
        f.write((char*)&m, 4);
        for (auto& p : sp_s2i) {
            uint32_t len = p.first.size();
            f.write((char*)&len, 4);
            f.write(p.first.data(), len);
            f.write((char*)&p.second, 4);
        }

        f.write((char*)&next_id, 4);
    }

    // 加载
    void load(const string& path) 
    {
        ifstream f(path, ios::binary);
        b2i.clear();
        sp_s2i.clear();
        sp_i2s.clear();

        uint32_t n;
        f.read((char*)&n, 4);
        for (uint32_t i = 0; i < n; ++i) 
        {
            uint32_t len;
            f.read((char*)&len, 4);
            vector<uint8_t> buf(len);
            f.read((char*)buf.data(), len);
            uint32_t id;
            f.read((char*)&id, 4);
            b2i[buf] = id;
        }

        uint32_t m;
        f.read((char*)&m, 4);
        for (uint32_t i = 0; i < m; ++i) 
        {
            uint32_t len;
            f.read((char*)&len, 4);
            string s(len, 0);
            f.read((char*)s.data(), len);
            uint32_t id;
            f.read((char*)&id, 4);
            sp_s2i[s] = id;
            sp_i2s[id] = s;
        }

        f.read((char*)&next_id, 4);
        rebuild_i2b();
    }

    uint32_t vocab_size() { return next_id; }
};

// ===================== 测试示例 =====================
int main2() {
    // 训练语料
    vector<string> corpus = {
        "用电电电鳗电鳗会不会被电电死?",
       // "一生一代一双人",
       // "Hello world",
       // "今天天气不错",
       // "BPE tokenizer test",
       // "你好世界"
    };

    // 训练
    BPETokenizer tok;
    tok.train(corpus, 800);

    // 特殊 token
    tok.add_special_tokens({ "<|im_start|>", "<|im_end|>", "<|endoftext|>" });

    // 保存 & 加载
    tok.save("tokenizer.bin");
    BPETokenizer tok2;
    tok2.load("tokenizer.bin");

    // 编码
    string test = "<|im_start|>用电电电鳗电鳗会不会被电电死<|im_end|>";
    auto [ids, tokens] = tok2.encode(test);

    cout << "encode ids: ";
    for (auto id : ids) cout << id << " ";
    cout << endl;

    // 解码
    string res = tok2.decode(ids);
    cout << "decode: " << res << endl;

    return 0;
}








int main()
{
    vector<string> corpus = 
    {
        "用电电电鳗电鳗会不会被电电死?",
        "bbpe 是 byte level bpe 分词算法。",
        "bpe 算法用于大模型 token 编码。",
        "bbpe 基于 utf8 字节合并中文英文。",
        "token 编码电鳗放电测试。",
        "token  to a ab abc abc  abcd abcf.,，。"
    };

    BBPE bbpe;
    bbpe.Train(corpus);

   /// auto ss = ToUTF8(a);
  //  main2();
#if 1
	//autogradMain();
	//CnnMain();
	//RnnMain();

	//EmbeddingMain();
	/// ResNetMain();

	//TransformerMain();

	//TransformerAttentionMain();
	///HandwrittenTransformerMain();
	//  DecoderOnlyMain();
	  //std::cout << "DecoderOnlyMain ...." << std::endl;
#else

	NetModule net;
	double learning_rate = 0.5;
	//torch::Device device(torch::kCPU);
	//net.to(device);
	torch::nn::MSELoss funloss;
	torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(learning_rate));
	auto input = torch::tensor({ 0.050,0.10 }, torch::kDouble);
	auto labels = torch::tensor({ 0.10,0.99 }, torch::kDouble);

	int64_t epochs = 10000 * 30;

	double accuracy = 0.0000006;

	auto start_time = chrono::high_resolution_clock::now();

	for (int64_t epoch = 0; epoch < epochs; ++epoch)
	{
		auto out = net.forward(input);
		auto loss = funloss(out, labels);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		if (abs(out.index({ 0 }).item<double>() - labels.index({ 0 }).item<double>()) <= accuracy &&
			abs(out.index({ 1 }).item<double>() - labels.index({ 1 }).item<double>()) <= accuracy)
		{
			std::cout << " break [" << epoch + 1 << "/" << epochs << "], Loss: " << loss.item<double>() << ", out " << out << std::endl;
			break;
		}


		if ((epoch + 1) % 100 == 0)
		{
			std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Loss: " << loss << std::endl;
		}

	}

	auto end_time = chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	std::cout << "end-time " << duration_ms << endl;


	cout << "<<<-------------------------------------------------->>>" << endl;
	cout << net.fc1->weight << endl << net.fc1->bias << endl;
	cout << "-----------------------------------------------------" << endl;
	cout << net.fc2->weight << endl << net.fc2->bias << endl;
	cout << "<<<<-------------------------------------------------- >>>" << endl << endl;

	
	
#endif

	std::cin.get();
	return 0;
}

/*
*

Epoch [100/300000], Loss: 0.004742825631
[ CPUDoubleType{} ]
Epoch [200/300000], Loss: 7.291057732e-07
[ CPUDoubleType{} ]
Epoch [300/300000], Loss: 2.949105029e-10
[ CPUDoubleType{} ]
 break [393/300000], Loss: 0.0000000000, out  0.1000   0.9900
[ CPUDoubleType{2} ]

end-time 2943
*/


