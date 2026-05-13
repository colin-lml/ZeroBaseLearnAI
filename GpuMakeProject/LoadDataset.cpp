
#include "LoadDataset.h"
#include "DecodersOnly.h"
#include <filesystem>
//#include <torch/optim.h>
//#include <torch/optim/schedulers.h>  


int64_t  gBOS = 0;
int64_t  gEOS = 0;
int64_t  gPad = 0;
int64_t  gVocabCount = 0;
size_t   m_gMaxBatch = 0;

torch::DeviceType gDType = torch::kCPU;

#define LR   (8e-5)


vector<vector<int64_t>> MakeTestData(int count)
{
    count = min(count, 50);

    gVocabCount = 55;

    gBOS = 51;
    gEOS = 52;
    gPad = 53;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 49);
    vector<vector<int64_t>> data;
    for (size_t i = 0; i < count; i++)
    {
        vector<int64_t> item;
    
        for (int j = 1; j < 50 -  i; j++)
        {
            int randomNumber =  i + j;//dis(gen);
            item.push_back(randomNumber);
          
        }
        
        data.push_back(item);

    }

    return data;
}

torch::optim::Adam CreateOptimizer(DecodersOnly& model)
{
    torch::optim::AdamOptions opt(LR);
    opt.betas({ 0.9, 0.98 });
    opt.eps(1e-9);
    opt.weight_decay(1e-8);
 
    return torch::optim::Adam(model->parameters(), opt);
}


void  LoadTrainState(const string& path, DecodersOnly& model, torch::optim::Adam& optimizer, int& step)
{
    
    step = 0;
    std::ifstream f(path);
    bool exists = f.good();

    if (exists)
    {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        model->load(archive);
        model->to(gDType);
       // optimizer = CreateOptimizer(model);
        optimizer.load(archive);
        
        c10::IValue s = 0;
        archive.read("step", s);
        step = s.toInt();
      
    }
    
}

void SaveTrainState(const string& path, DecodersOnly& model, torch::optim::Adam& optimizer, int step)
{
    
    torch::serialize::OutputArchive archive;

    model->save(archive);
    optimizer.save(archive);
    archive.write("step", step);

    archive.save_to(path);
}

void  SaveModel(DecodersOnly& model, const string& path)
{
    torch::serialize::OutputArchive archive;
    model->save(archive);
  
    archive.save_to(path);
}
void  LoadModel(DecodersOnly& model, const string& path)
{
    
    std::ifstream f(path);
    bool exists = f.good();
    f.close();

    if (exists)
    {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        model->load(archive);
        model->to(gDType);
    }
}


void TrainData(DecodersOnly& model, translatDatasetOnly& dataTrain, int64_t maxtrain, int64_t batchsize)
{
 
    auto p = std::filesystem::current_path().string();
    string strCheckpoint = "acpu.model.checkpoint.pt";
    string  modelTmpPath = "";
    double accuracy = 0.06;
    auto options = torch::nn::CrossEntropyLossOptions().ignore_index(gPad);
    torch::nn::CrossEntropyLoss loss_fn(options);

     

    torch::optim::Adam optimizer = CreateOptimizer(model);
    
   // torch::optim::ReduceLROnPlateauScheduler lr_scheduler(optimizer);

    BatchSampler sampler(&dataTrain);

    auto datasetTrain = dataTrain.map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader(std::move(datasetTrain), std::move(sampler),torch::data::DataLoaderOptions().batch_size(batchsize));

    int step = 0;
   

    std::cout << "训练模型, " ;

    model->train();
    string logtrain = p + "/../tmpbin/aCPU.train.log";
    int showItem = 10;
    if (gDType == torch::kCUDA)
    {
        showItem = 20;
        logtrain = p + "/../tmpbin/acuda.train.log";
        strCheckpoint =  "aCUDA.model.checkpoint.pt";
        modelTmpPath = "Decoder_Only_model3.pt.cuda";
    }
    else
    {
        modelTmpPath = "Decoder_Only_model3.pt.cpu";
    }

    LoadTrainState(strCheckpoint, model, optimizer, step);


    std::cout << " step: "<< step << std::endl;

    ofstream logFile(logtrain, step == 0 ? ios::out : ios::app);

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
       // lr_scheduler.step(total_loss);
        if (std::isnan(total_loss))
        {
            cout <<"########### nan break : " << i + 1 << endl;
            break;
        }


        if (i % showItem == 0 || (i + 1) == maxtrain)
        {
            cout << i + 1 <<" / "<< maxtrain << " , total-loss: " << total_loss << " , loss: " << loss1 << endl;
        
            logFile << i + 1 << " / " << maxtrain << " , total-loss: " << total_loss << " , loss: " << loss1 << endl;
        }

        if (total_loss < accuracy)
        {
            cout << i + 1 << " / " << maxtrain << " , loss: " << total_loss << " , loss: " << loss1 << " , end... " << endl;
            logFile << i + 1 << " / " << maxtrain << " , loss: " << total_loss << " , loss: " << loss1 << " , end... " << endl;
            break;
        }

        if (i % (showItem/2) == 0)
        {
            SaveTrainState(strCheckpoint, model, optimizer, i);
            SaveModel(model, modelTmpPath);
        }

    }
    logFile.close();
}



void TestData(DecodersOnly& model, translatDatasetOnly& dataTest)
{
    model->eval();
    std::cout << "测试:" << std::endl;
    std::vector<std::string> tests;

    tests.push_back("春眠不觉晓");
    tests.push_back("床前明月光");
    tests.push_back("白日依山尽");
    tests.push_back("空山不见人");

    for (auto ch : tests)
    {
        auto result = model->predict(ch, dataTest);

        // std::cout << std::regex_replace(ch, std::regex("Pad"), "") << " :  ";

        std::cout << ch << " :" << std::endl;
        std::cout << result << std::endl;;

        std::cout << std::endl;
    }

    do 
    {
        string line;
        std::cout << "input: ";
        getline(cin, line);
        if (line=="exit" || line == "e")
        {
            break;
        }
    
        auto result = model->predict(line, dataTest);
        std::cout <<"\noutput:\n" << result << std::endl << std::endl;

    } while (true);
    

  
}


