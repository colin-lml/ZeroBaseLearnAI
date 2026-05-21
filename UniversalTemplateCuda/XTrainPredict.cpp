#include "pch.h"
#include "XTrainPredict.h"



XTrainPredict::XTrainPredict()
{
	m_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	string device = (m_device == torch::kCUDA) ? "kCUDA" : "kCPU";
	cout << "device type: " << device << endl;

    
    if (m_device == torch::kCUDA)
    {
        m_strLogTrain = "cuda.train.log";
        m_strCheckpoint = "cuda.checkpoint.pt";
        m_strTmpModelPath = m_strModelPath + ".cuda";
    }
    else
    {
        m_strLogTrain = "cpu.train.log";
        m_strCheckpoint = "cpu.checkpoint.pt";
        m_strTmpModelPath = m_strModelPath + ".cpu";
    }

}

void XTrainPredict::TestData()
{

    int64_t numword = m_xDataset.GetDictionarySize();
    int64_t pad = m_xDataset.GetPAD();

    XDecoderOnly model(m_numHeads, numword, m_numLayers, pad);

    int64_t totalParams = 0;
    for (const auto& p : model->parameters())
    {
        totalParams += p.numel();
    }

  
    cout << "model total params: " << totalParams << endl;

    if (!LoadModel(model))
    {
        if (TrainData(model))
        {
            SaveModel(model, m_strModelPath);
        }
    }
    std::cout << "测试:" << std::endl;
    model->eval();
    std::vector<std::string> tests;

    tests.push_back("春眠不觉晓");
    tests.push_back("床前明月光");
    tests.push_back("白日依山尽");
    tests.push_back("空山不见人");

    VectorInt64 vList;
    int64_t eos = m_xDataset.GetEOS();
    int64_t bos = m_xDataset.GetBOS();

    for (auto& ch : tests)
    {
        m_xDataset.Encode(ch, vList);
        vList.insert(vList.begin(), bos);
        model->predict(m_device,vList, eos, 50);

       auto str =  m_xDataset.Decoded(vList);
       cout <<"input: "<< ch<<"\n" << str << endl << endl;

    }

    do
    {
        string line;
        std::cout << "input: ";
        getline(cin, line);
        if (line == "exit" || line == "e")
        {
            break;
        }

        m_xDataset.Encode(line, vList);
        vList.insert(vList.begin(), bos);
        model->predict(m_device, vList, eos, 50);

        auto str = m_xDataset.Decoded(vList);
        cout  << str << endl << endl;

    } while (true);

}

bool XTrainPredict::TrainData(XDecoderOnly& model)
{

    bool b= true;
    double accuracy = 0.06;
    auto options = torch::nn::CrossEntropyLossOptions().ignore_index(m_xDataset.GetPAD());
    torch::nn::CrossEntropyLoss lossFnCEL(options);
    torch::optim::Adam optimizer = CreateOptimizer(model);
    int step = 0;

    model->to(m_device);
    LoadTrainingBreakpoint(model, optimizer, step);

    LogStream log(m_strLogTrain, step == 0 ? ios::out : ios::app);
    log << "batchsize: " << m_batchsize<<" , LR: " << LR<< " , head: "<< m_numHeads<<" , layer: " << m_numLayers << " , dim: " << (m_numHeads * 64) << std::endl;
    log << "训练模型,  step: " << step << std::endl;

    XBatchSampler sampler(*m_xDataset.size());
    auto datasetTrain = m_xDataset.map(torch::data::transforms::Stack<>());
    auto trainDataLoader = torch::data::make_data_loader(std::move(datasetTrain), std::move(sampler), torch::data::DataLoaderOptions().batch_size(m_batchsize));
    
    int64_t checkpoint = 40;

    model->train();

    for (int64_t i = step; i < m_maxtrain; i++)
    {
        float totalLoss = 0;
        float sinLoss = 0;
        for (auto& item : *trainDataLoader)
        {
            optimizer.zero_grad();
           
            auto output = model->forward(item.data.to(m_device));

            output = output.reshape({ -1, output.size(2) });
            auto tgt = item.target.to(m_device).reshape({ -1 });
            auto loss = lossFnCEL(output, tgt);

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            loss.backward();
            optimizer.step();
            sinLoss = loss.item<float>();
            totalLoss += sinLoss;
        }

        if (std::isnan(totalLoss))
        {
            log << "########### nan break : " << i + 1 << endl;
           
            b = false;
            
            break;
        }

        if (totalLoss < accuracy)
        {
            log << i << " / " << m_maxtrain << " ,totalLoss: " << totalLoss << ", singleLoss: " << sinLoss << std::endl;
            break;
        }

        if (i % checkpoint == 0)
        {
            SaveTrainingBreakpoint(model, optimizer, i);

            if (totalLoss < 0.9)
            {
                SaveModel(model, m_strTmpModelPath);
            }

            log << i<<" / " << m_maxtrain << " ,totalLoss: " << totalLoss << ", singleLoss: " << sinLoss << std::endl;
        }

    }

    return b;
}

torch::optim::Adam XTrainPredict::CreateOptimizer(XDecoderOnly& model)
{
    torch::optim::AdamOptions opt(LR);
    opt.betas({ 0.9, 0.98 });
    opt.eps(1e-9);
    opt.weight_decay(0);

    return torch::optim::Adam(model->parameters(), opt);
}



void  XTrainPredict::SaveModel(XDecoderOnly& model, const string& path)
{
    auto binPath = GetOutputPath() + path;
    torch::serialize::OutputArchive archive;
    model->save(archive);

    archive.save_to(binPath);
}

bool XTrainPredict::LoadModel(XDecoderOnly& model)
{
    auto binPath = GetOutputPath() + m_strModelPath;
    std::ifstream f(binPath);
    bool exists = f.good();
    f.close();

    if (exists)
    {
        torch::serialize::InputArchive archive;
        archive.load_from(binPath);
        model->load(archive);
        model->to(m_device);
    }

    return exists;
}

void XTrainPredict::LoadTrainingBreakpoint(XDecoderOnly& model, torch::optim::Adam& optimizer, int& step)
{
    auto binPath = GetOutputPath() + m_strCheckpoint;
    step = 0;
    std::ifstream f(binPath);
    bool exists = f.good();

    if (exists)
    {
        torch::serialize::InputArchive archive;
        archive.load_from(binPath);
        model->load(archive);
        model->to(m_device);
        optimizer.load(archive);

        c10::IValue s = 0;
        archive.read("step", s);
        step = s.toInt();

    }
}

void XTrainPredict::SaveTrainingBreakpoint(XDecoderOnly& model, torch::optim::Adam& optimizer, int step)
{
    auto binPath = GetOutputPath() + m_strCheckpoint;
    torch::serialize::OutputArchive archive;

    model->save(archive);
    optimizer.save(archive);
    archive.write("step", step);

    archive.save_to(binPath);
}
