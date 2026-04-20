#include "Tokenizer.h"
#include <windows.h>
#include <sstream>

void Tokenizer::LoadDataSrc()
{

    LoadDataTxtFile();

    InitTokenizer(m_vdata);

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
    for each(auto& item in stringList)
    {
        if (stringID.find(item) == stringID.end())
        {
            int64_t index = stringID.size();
            stringID.emplace(item, index);
            IDString.emplace(index, item);
        }
    }
}
