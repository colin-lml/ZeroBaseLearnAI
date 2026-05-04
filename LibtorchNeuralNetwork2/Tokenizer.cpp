#include "Tokenizer.h"
#include <windows.h>
#include <sstream>

void Tokenizer::InitLoadDataSrc()
{
    bool b = loadMap();
    if (!b)
    {
        LoadDataTxtFile();
        InitTokenizer(m_vdata);

        InitEncodeTangshi(m_vdata);

        saveMap(m_vEncodeDataList);
    }

    /* 
    for (auto& i : m_vEncodeDataList)
    {
        auto str = m_bbpe.Decode(i.title);
        auto str2 = m_bbpe.Decode(i.author);
        auto str3 = m_bbpe.Decode(i.content);
        std::cout<< str << std::endl;
        std::cout << str2 << std::endl;
        std::cout << str3 << std::endl;
    }
   */
}

void Tokenizer::LoadDataTxtFile()
{
    m_vdata.clear();

	std::ifstream ifs("tangshi.data.txt");
	bool bopen = ifs.is_open();
    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    std::string line;

    while (true)
    {
        while (getline(ss, line) && line.empty());
        if (!ss)
        {
            break;
        }

        Tangshi item;

        std::string n;
        std::stringstream data(line);
        data >>  item.title;
   
        if (getline(ss, line))
        {
            item.author = line;  
        }

        while (getline(ss, line))
        {
            if (line.empty()) 
            {
                break;
            }
            item.content += line + "\n";
        }
        m_vdata.push_back(item);
    }
}


void Tokenizer::InitTokenizer(std::vector<Tangshi>& vDataList)
{
    VectorString k;
    for each(auto& item in vDataList)
    {
        k.push_back(item.title);
        k.push_back(item.author);
        k.push_back(item.content);
    }

    m_bbpe.Train(k);

}


void Tokenizer::saveMap(std::vector<VectorCodeTangshi>& vData)
{
    std::ofstream ofs(m_strBinFile, std::ios::binary);
    size_t count = 0;//map1.size();

    count = vData.size();
    ofs.write((const char*)&count, sizeof(count));

    for (auto& vCode : vData)
    {
        size_t len = vCode.title.size();
        ofs.write((const char*)&len, sizeof(len));
        ofs.write((const char*)vCode.title.data(), len * sizeof(int64_t));

        len = vCode.author.size();
        ofs.write((const char*)&len, sizeof(len));
        ofs.write((const char*)vCode.author.data(), len * sizeof(int64_t));

        len = vCode.content.size();
        ofs.write((const char*)&len, sizeof(len));
        ofs.write((const char*)vCode.content.data(), len * sizeof(int64_t));
    }

    ofs.close();
}

bool Tokenizer::loadMap()
{
    bool b = false;

    std::ifstream ifs(m_strBinFile, std::ios::binary);
    if (!ifs)
    {
        return b;
    }
 
    size_t count = 0;

    m_vEncodeDataList.clear();

    ifs.read((char*)&count, sizeof(count));

    for (size_t i = 0; i < count; ++i) 
    {
        VectorCodeTangshi item;

        size_t len = 0;
        ifs.read((char*)&len, sizeof(len));
        item.title.resize(len);
        ifs.read((char*)item.title.data(), len * sizeof(int64_t));

        ifs.read((char*)&len, sizeof(len));
        item.author.resize(len);
        ifs.read((char*)item.author.data(), len * sizeof(int64_t));
  
        ifs.read((char*)&len, sizeof(len));
        item.content.resize(len);
        ifs.read((char*)item.content.data(), len * sizeof(int64_t));

        m_vEncodeDataList.push_back(item);
    }

    ifs.close();
    b = true;

    return b;
}


void  Tokenizer::InitEncodeTangshi(std::vector<Tangshi>& vDataList)
{
    m_vEncodeDataList.clear();

    for (auto& item : vDataList)
    {
        VectorCodeTangshi codeid;
        m_bbpe.Encode(item.title, codeid.title);
        m_bbpe.Encode(item.author, codeid.author);
        m_bbpe.Encode(item.content, codeid.content);
        m_vEncodeDataList.push_back(codeid);
    }
}
