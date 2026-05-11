
#include "LoadDataset.h"
#include "DecodersOnly.h"
#include <filesystem>

int64_t  gBOS = 0;
int64_t  gEOS = 0;
int64_t  gPad = 0;
int64_t  gVocabCount = 0;

torch::DeviceType gDType = torch::kCPU;


vector<pair<vector<int64_t>, vector<int64_t>>> MakeTestData(int count)
{

    count = min(count, 50);

    gVocabCount = 55;
    gBOS = count + 1;
    gEOS = count + 2;
    gPad = count + 3;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 49);
    vector<pair<vector<int64_t>, vector<int64_t>>> data;
    for (size_t i = 0; i < count; i++)
    {
        vector<int64_t> in;
        vector<int64_t> lab;
        in.push_back(gBOS);

        for (int j = 1; j < 55 -  i; j++)
        {
            int randomNumber =  i + j;//dis(gen);
            in.push_back(randomNumber);
          
        }
        
        

        for (int k = 0; k < i; k++)
        {
            in.push_back(gPad);
        }
        //in.push_back(gEOS);

        lab = in;
        lab.erase(lab.begin());
        lab.push_back(gEOS);
        
        data.push_back({ in ,lab });

    }

    return data;
}


void  LoadTrainState(const string& path, const string& path2, DecodersOnly& mode, torch::optim::Adam& optimizer, int& step)
{
    /*
    step = 0;
    ifstream f(path2, ios::binary);
    if (f.is_open())
    {
        f.read((char*)&step, sizeof(step));

        torch::OrderedDict<std::string, torch::IValue> dict;
        torch::load(dict, path);
        mode = dict["model"].to<DecodersOnly>();
        optimizer = dict["optim"].to<torch::optim::Adam>();
    }
    */
}

void SaveTrainState(const string& path, const string& path2, DecodersOnly& mode, torch::optim::Adam& optimizer, int step)
{
    /*
    torch::OrderedDict<std::string, torch::IValue> dict;
    dict.insert("model", mode);
    dict.insert("optim", optimizer);
    torch::save(dict, path);

    ofstream f(path2, ios::binary);
    f.write((char*)&step, sizeof(step));
    */
}


void TrainData(DecodersOnly& model, translatDatasetOnly& dataTrain, int64_t maxtrain, int64_t batchsize)
{
    auto p = std::filesystem::current_path().string() + "/../Decoder_Only_model3_tmp.pt";
    string strTmpState = "TrainData.tmp.pt.bin";
    string strTmpState2 = "TrainData.tmp.step.bin";
    double accuracy = 0.05;
    auto options = torch::nn::CrossEntropyLossOptions().ignore_index(gPad);
    torch::nn::CrossEntropyLoss loss_fn(options);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    auto datasetTrain = dataTrain.map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), torch::data::DataLoaderOptions().batch_size(batchsize));

    int step = 0;
    // LoadTrainState(strTmpState, strTmpState2, model, optimizer, step);

    std::cout << "ÑµÁ·Ä£ÐÍ" << std::endl;

    model->train();

    int showItem = 10;
    int showSave = 1000;
    if (gDType == torch::kCUDA)
    {
        showItem = 20;
        showSave = 500;
    }

    for (int i = step; i < maxtrain; i++)
    {
        float total_loss = 0;
        float loss1 = 0;
        for (auto& item : *train_data_loader)
        {   
  
            auto output = model->forward(item.data.to(gDType));
          
            output = output.reshape({ -1, output.size(2) });
            auto tgt = item.target.to(gDType).reshape({ -1 });

            auto loss = loss_fn(output, tgt);

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();
            loss1 = loss.item<float>();
            total_loss += loss1;

        }
       
        if (std::isnan(total_loss))
        {
            cout <<"########### nan break : " << i + 1 << endl;
            break;
        }


        if (i % showItem == 0 || (i + 1) == maxtrain)
        {
            cout << i + 1 <<" / "<< maxtrain << " , total-loss: " << total_loss << " , loss: " << loss1 << endl;
        }

        if (total_loss < accuracy)
        {
            cout << i + 1 << " / " << maxtrain << " , loss: " << total_loss << " , loss: " << loss1 << " , end... " << endl;
            break;
        }
        if (i % showSave == 0 && total_loss < 1.2)
        {
            // SaveTrainState(strTmpState, strTmpState2, model, optimizer, i);
           
            torch::save(model, p);
        }

    }

    torch::save(model, p);

    std::remove(strTmpState2.c_str());
    std::remove(strTmpState.c_str());
}



void TestData3(DecodersOnly& model, translatDatasetOnly& dataTest)
{
    model->eval();
    std::cout << "²âÊÔ:" << std::endl;
    std::vector<std::string> tests;

    tests.push_back("´ºÃß²»¾õÏþ");
    //tests.push_back("°×ÈÕÒÀÉ½¾¡");

    for (auto ch : tests)
    {
        auto result = model->predict(ch, dataTest);

        // std::cout << std::regex_replace(ch, std::regex("Pad"), "") << " :  ";

        std::cout << ch << " :" << std::endl;
        std::cout << result << std::endl;;

        std::cout << std::endl;
    }

}


