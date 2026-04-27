#include "BBPE.h"

void remove_all_spaces(std::string& s)
{
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
}
bool isDelimiter(uint8_t  c)
{
    return c == ' ' || c == '£¬' || c == '¡£' ;
}


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

VectorString BBPE::SplitText(const string& str)
{
    VectorString  wlist;
    string current;

    for (auto& ch : str)
    {
        if (isDelimiter(ch))
        {
            if (!current.empty()) 
            {
                wlist.push_back(current);
                current.clear();
            }
        }
        else
        {
            current += ch;
        }
    }

    if (!current.empty())
    {
        wlist.push_back(current);
    }

    return wlist;
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
    Vector3Uint8 vEnumWordList;

  
    MapVocabTable historyMerge;
    
    EnumerationWord(textList, vEnumWordList);

    /* 
   for (auto& all : vEnumWordList)
   {
       for (auto &item : all)
       {
           string strutf8(item.begin(), item.end());
           string gbk = ToGBK(strutf8);
           cout << gbk;
       }
       cout << endl;
   }
   */

    while ((historyMerge.size() + m_mapVocabTable.size()) < vocabSize)
    {
        MapVocabTable wordCount;
        wordCount.clear();
        CountPairWord(vEnumWordList, wordCount);

        auto[pairKey,pairCount] = FindMaxPairCount(wordCount);

        if (wordCount.empty() || pairCount == 0)
        {
            break;
        }

        MergeMaxPairWord(vEnumWordList, historyMerge, pairKey);    
    }

    for (auto& x : historyMerge)
    {
        auto item = x.first;
        string strutf8(item.begin(), item.end());
        string gbk = ToGBK(strutf8);
        cout << gbk << endl;
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

void BBPE::EnumerationWord(const VectorString& textList, Vector3Uint8& vEnumWordList)
{
    vEnumWordList.clear();
    VectorString vAllString;

    for (auto& slist : textList)
    {
       auto  item = SplitText(slist);
       vAllString.insert(vAllString.end(), item.begin(), item.end());
    }
     
    for (auto& slist : vAllString)
    {
        auto strutf8 =  ToUTF8(slist);

        Vector2Uint8 item;
        for (int i=0;i< strutf8.size();i++)
        {
            int len = GetWordSzie(strutf8[i]);
            VectorUint8 word;

            for (int j = 0; j < len; j++)
            {
                word.push_back(strutf8[i + j]);
            }
            item.push_back(word);

            i += len-1;
        }

        vEnumWordList.push_back(item );
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

void BBPE::MergeWord(VectorUint8& outMerge, const VectorUint8& a, const VectorUint8& b)
{
    auto k = a.size();
    auto m = b.size();

    outMerge.reserve(k + m);

    outMerge.insert(outMerge.end(), a.begin(), a.end());
    outMerge.insert(outMerge.end(), b.begin(), b.end());
}

void BBPE::CountPairWord(Vector3Uint8& vAllWordList, MapVocabTable& vPairCount)
{
    for (int i = 0; i < vAllWordList.size(); i++)
    {
        auto& items = vAllWordList[i];

        for (int j = 0; j+1 < items.size(); j++)
        {
            VectorUint8 merged;

            MergeWord(merged, items[j], items[j + 1]);

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
}

void BBPE::MergeMaxPairWord(Vector3Uint8& vAllWordList, MapVocabTable& historyMerge, VectorUint8& tgtKey)
{
    Vector3Uint8 temp;

    for (int i = 0; i < vAllWordList.size(); i++)
    {
        auto& items = vAllWordList[i];
        Vector2Uint8 tempItem;
        for (int j = 0; j+1 < items.size(); j++)
        {
            VectorUint8 merged;
            MergeWord(merged, items[j], items[j + 1]);

            if (merged == tgtKey)
            {
                tempItem.push_back(merged);
                j += 1;
                historyMerge.emplace(merged, 1);
            }
            else
            {
                tempItem.push_back(items[j]);
            }
        }
        temp.push_back(tempItem);
    }

    vAllWordList.swap(temp);
}