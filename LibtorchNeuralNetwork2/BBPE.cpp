#include "BBPE.h"


BBPE::BBPE()
{
    InitData();
}
BBPE::~BBPE()
{

}

string  BBPE::ToUTF8(const string& str)
{
    return MultiByteToMultiByte(str);
}

string  BBPE::ToGBK(const string& str)
{
    return MultiByteToMultiByte(str, CP_UTF8, CP_ACP);
}

string BBPE::MultiByteToMultiByte(const string& str, UINT from , UINT bto )
{
    int wide_size = MultiByteToWideChar(from, 0, str.c_str(), -1, NULL, 0);

    std::wstring wideStr(wide_size, 0);
    MultiByteToWideChar(from, 0, str.c_str(), -1, wideStr.data(), wide_size);

    int utf8_size = WideCharToMultiByte(bto, 0, wideStr.data(), -1, NULL, 0, NULL, NULL);

    std::string multiStr(utf8_size, 0);

    WideCharToMultiByte(bto, 0, wideStr.data(), -1, multiStr.data(), utf8_size, NULL, NULL);
    multiStr.pop_back();

    return multiStr;
}


void BBPE::Train(const VectorString& textList, uint32_t vocabSize)
{
    VectorVectorUint vEnumWordList;
    
    EnumerationWord(textList, vEnumWordList);

    while (m_mapVocabTable.size() < vocabSize)
    {
        MapVocabTable wordCount;
        wordCount.clear();
        CountPairWord(vEnumWordList, wordCount);

        auto[pairKey,pairCount] = FindMaxPairCount(wordCount);

        if (wordCount.empty() || pairCount == 0)
        {
            break;
        }



    }

   
    /*
    for (auto& item : vEnumWordList)
    {
        string strutf8(item.begin(), item.end());
        string gbk= ToGBK(strutf8);
        cout << gbk << endl;
    }   
    */
}

bool BBPE::IsExistVocabTable(VectorUint8& v)
{
    return m_mapVocabTable.find(v) != m_mapVocabTable.end();
}

void BBPE::AddNewKeyToVocabTable(VectorUint8& vlist)
{
    auto id = m_mapVocabTable.size();

    if (!IsExistVocabTable(vlist))
    {
        m_mapVocabTable.emplace(vlist, id);
        m_mapIDtoCodeId.emplace(id, vlist);
    }
}

void BBPE::InitData()
{
    m_mapVocabTable.clear();
    m_mapIDtoCodeId.clear();
   
    for (int i = 0; i < 256; i++)
    {
        VectorUint8 b;
        b.push_back(i);
        AddNewKeyToVocabTable(b);
    }
}

int  BBPE::GetWordSzie(uint8_t ch)
{
    int len = 1;
    if ((ch & 0x80) == 0)
    {
        len = 1; // ASCII
    }
    else if((ch & 0xE0) == 0xC0)
    {
        len = 2;
    }
    else if ((ch & 0xF0) == 0xE0)
    {
        len = 3; // ÖÐÎÄ
    }
    else if ((ch & 0xF8) == 0xF0)
    {
        len = 4;
    }
    else
    {
        len = 1;
    }

    return  len;
}

void BBPE::EnumerationWord(const VectorString& textList, VectorVectorUint& vEnumWordList)
{
    vEnumWordList.clear();

    for (auto& slist : textList)
    {
        auto strutf8 =  ToUTF8(slist);
        for (int i=0;i< strutf8.size();i++)
        {
            int len = GetWordSzie(strutf8[i]);
            VectorUint8 word;
            for (int j = 0; j < len; j++)
            {
                word.push_back(strutf8[i+j]);
            }

            vEnumWordList.push_back(word);

            i += len-1;
        }
    }

}

pair<VectorUint8,int64_t> BBPE::FindMaxPairCount(MapVocabTable& vPairCount)
{
    VectorUint8 key;
    int64_t count = 0;

    for (auto& p : vPairCount)
    {
        if (p.second > count)
        {
            count = p.second;
            key = p.first;
        }
    }
    
    return {key, count };
}


void BBPE::CountPairWord(VectorVectorUint& vAllWordList, MapVocabTable& vPairCount)
{
    for (int i = 0; i < vAllWordList.size()-1; i++)
    {
        VectorUint8 merged;
        auto a = vAllWordList[i].size();
        auto b = vAllWordList[i + 1].size();

        merged.reserve(a + b);

        merged.insert(merged.end(), vAllWordList[i].begin(), vAllWordList[i].end());
        merged.insert(merged.end(), vAllWordList[i + 1].begin(), vAllWordList[i + 1].end());
        
        if (vPairCount.find(merged) == vPairCount.end())
        {
            vPairCount.emplace(merged, 0);
        }
        else
        {
            vPairCount[merged]++;
        }
        
    }
}

void MergeMaxPairWord()
{
}