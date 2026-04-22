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

        saveMap(m_stringToID, m_vEncodeDataList);
    }
    /* 
    for (auto& i : m_vEncodeDataList)
    {
        auto str = GetTangshiString(i);
        std::cout<< str << std::endl;
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
        data >> n >> item.title;
   
        m_nMaxTitle = std::max(m_nMaxTitle, ChineseCount(item.title));

        if (getline(ss, line))
        {
            item.author = line;
            m_nMaxAuthor = std::max(m_nMaxAuthor, ChineseCount(item.author));
            if (m_nMaxAuthor == 4)
            {

            }
        }

        while (getline(ss, line))
        {
            if (line.empty()) 
            {
                break;
            }
            item.content += line + "\n";
        }

        m_nMaxContent = std::max(m_nMaxContent, ChineseCount(item.content));

        m_vdata.push_back(item);
    }
}

bool Tokenizer::IsChinese(const char& ch)
{
   return  (unsigned char)ch > 0x80;
}

int Tokenizer::ChineseCount(const std::string& s)
{
    int cnt = 0;
    for (int i = 0; i < s.size(); i++) 
    {
        if (IsChinese(s[i]))
        {
            i++; 
        }
        cnt++;
    }
    return cnt;
}

std::vector<std::string> Tokenizer::SplitString(std::string line)
{
    std::vector<std::string> vData;
    vData.clear();
    char arrCh[3] = { 0 };

    for (size_t i = 0; i < line.length(); i++)
    {
        char arrCh[3] = { 0 };
        arrCh[0] = line[i];
        if (IsChinese(line[i]))
        {
            i++;
            arrCh[1] = line[i];
        }
        vData.push_back(arrCh);
    }
    return vData;
}


void Tokenizer::InitTokenizer(std::vector<Tangshi>& vDataList)
{

    m_stringToID.clear();
    m_IDToString.clear();

    m_stringToID.emplace("Pad",0);
    m_stringToID.emplace("S", 1);
    m_stringToID.emplace("E", 2);

    m_IDToString.emplace(0, "Pad");
    m_IDToString.emplace(1, "S");
    m_IDToString.emplace(2, "E");

    for each(auto& item in vDataList)
    {
        auto vt = SplitString(item.title);
        auto va = SplitString(item.author);
        auto vc = SplitString(item.content);

        AddVocabTable(vt, m_stringToID, m_IDToString);
        AddVocabTable(va, m_stringToID, m_IDToString);
        AddVocabTable(vc, m_stringToID, m_IDToString);
    }



}

void Tokenizer::AddVocabTable(std::vector<std::string>& stringList, CorpusVocabStoID& stringID, CorpusVocabIDtoS& IDString)
{
    for(auto& item : stringList)
    {
        if (stringID.find(item) == stringID.end())
        {
            int64_t index = stringID.size();
            stringID.emplace(item, index);
            IDString.emplace(index, item);
        }
    }
}

void Tokenizer::saveMap(CorpusVocabStoID& map1, std::vector<std::vector<int64_t>>& vData)
{
    std::ofstream ofs(m_strBinFile, std::ios::binary);
    size_t count = map1.size();

    ofs.write((const char*)&m_nMaxTitle, sizeof(m_nMaxTitle));
    ofs.write((const char*)&m_nMaxAuthor, sizeof(m_nMaxAuthor));
    ofs.write((const char*)&m_nMaxContent, sizeof(m_nMaxContent));

    ofs.write((const char*)&count, sizeof(count));

    for (auto& pair : map1)
    {
        size_t len = pair.first.size();
        ofs.write((const char*)&len, sizeof(len));
        ofs.write(pair.first.c_str(), len);

        ofs.write((const char*)&pair.second, sizeof(int64_t));
    }

    count = vData.size();
    ofs.write((const char*)&count, sizeof(count));

    for (auto& vCode : vData)
    {
        size_t len = vCode.size();
        ofs.write((const char*)&len, sizeof(len));
        ofs.write((const char*)vCode.data(), len * sizeof(int64_t));
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

    m_stringToID.clear();
    m_IDToString.clear();

    size_t count = 0;
    ifs.read((char*)&m_nMaxTitle, sizeof(m_nMaxTitle));
    ifs.read((char*)&m_nMaxAuthor, sizeof(m_nMaxAuthor));
    ifs.read((char*)&m_nMaxContent, sizeof(m_nMaxContent));

    ifs.read((char*)&count, sizeof(count));

    for (size_t i = 0; i < count; ++i) 
    {
        size_t len = 0;
        ifs.read((char*)&len, sizeof(len));

        std::string strKey;
        strKey.resize(len);
        ifs.read(&strKey[0], len);

        int64_t id = 0;
        ifs.read((char*)&id, sizeof(int64_t));
        m_stringToID.emplace(strKey, id);
        m_IDToString.emplace(id, strKey);
       
    }

    m_vEncodeDataList.clear();
    count = 0;
    ifs.read((char*)&count, sizeof(count));
    for (size_t i = 0; i < count; ++i)
    {
        std::vector<int64_t> vec;
        size_t n = 0;
        ifs.read((char*)&n, sizeof(n));
        vec.resize(n);
        ifs.read((char*)vec.data(), n * sizeof(int64_t));
        m_vEncodeDataList.push_back(vec);
    }

    ifs.close();
    b = true;

    return b;
}

std::vector<int64_t> Tokenizer::GetTangshiCode(std::string& line)
{
    std::vector<int64_t> vData;
    auto vString = SplitString(line);
    for (auto& ch : vString)
    {
        if (m_stringToID.find(ch) != m_stringToID.end())
        {
            vData.push_back(m_stringToID.at(ch));
        }
    }
    return vData;
}

std::string Tokenizer::GetTangshiString(std::vector<int64_t>& vList)
{
    std::string line = "";

    for (auto& i : vList)
    {
        if (m_IDToString.find(i) != m_IDToString.end())
        {
            line += m_IDToString.at(i);
        }
    }

    return line;
}

void  Tokenizer::InitEncodeTangshi(std::vector<Tangshi>& vDataList)
{
    m_vEncodeDataList.clear();


    for (auto& item : vDataList)
    {
        std::vector<int64_t> encode;
        encode.push_back(1); // add S

        auto t = GetTangshiCode(item.title);
        auto a = GetTangshiCode(item.author);
        auto c = GetTangshiCode(item.content);

        int len = 0;

        len = m_nMaxContent - c.size() + m_nMaxAuthor - a.size() + m_nMaxTitle - t.size();
        for (int i = 0; i < len; i++)
        {
            c.push_back(0);
        }

        c.push_back(2); /// add E

        encode.insert(encode.end(), t.begin(), t.end());
        encode.insert(encode.end(), a.begin(), a.end());
        encode.insert(encode.end(), c.begin(), c.end());

        m_vEncodeDataList.push_back(encode);
    }
}
