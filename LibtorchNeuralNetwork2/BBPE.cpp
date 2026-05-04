#include "BBPE.h"

void remove_all_spaces(std::string& s)
{
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
}

BBPE::BBPE()
{
    if (!LoadFile())
    {
        InitData();
    }

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


vector<size_t> FindSplitIndex(const string& str, const string& d)
{
    size_t pos = 0, prev = 0;
    vector<size_t> vIndex;
   
    while ((pos = str.find(d, prev)) != std::string::npos)
    {
        vIndex.push_back(pos);
        prev = pos + d.size();
    }

    return vIndex;
}


string BBPE::VectorUint8ToGBK(const VectorUint8& item)
{
    string strutf8(item.begin(), item.end());
    string gbk = ToGBK(strutf8);
    return gbk;
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
    MapVocabPairCount historyMerge;

    std::remove(BBPE_PATH);

    InitData();
    DataCleansingWord(textList, vEnumWordList);


    while ((historyMerge.size() + m_mapVocabTable.size()) < vocabSize)
    {
        MapVocabPairCount wordCount;
        wordCount.clear();
        CountPairWord(vEnumWordList, wordCount);

        auto[pairKey,pairCount] = FindMaxPairCount(wordCount);

        if (wordCount.empty() || pairCount == 0)
        {
            break;
        }

        MergeMaxPairWord(vEnumWordList, historyMerge, pairKey,  pairCount);

    }
 
    for (auto& addItem : historyMerge)
    {
        AddNewKeyToVocabTable(addItem.first);
        //auto str = VectorUint8ToGBK(addItem.first);
        //cout << str<<" : "<< addItem.second <<endl;
    }

    VectorString special;
    special.push_back(BOS);
    special.push_back(EOS);
    special.push_back(PAD);
    //special.push_back();
    AddSpecialTokens(special);

    SaveFile();
    //LoadFile();
   
}

bool BBPE::IsExistVocabTable(const string& str)
{
    VectorUint8 v;
    for (int i = 0; i < str.length(); i++)
    {
        v.push_back(str[i]);
    }

    return IsExistVocabTable(v);
}



bool BBPE::IsExistVocabTable(const VectorUint8& v)
{
    return m_mapVocabTable.find(v) != m_mapVocabTable.end();
}

void BBPE::AddNewKeyToVocabTable(const VectorUint8& vlist)
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



void BBPE::DataCleansingWord(const VectorString& textList, Vector3Uint8& vEnumWordList)
{
    vEnumWordList.clear();
    VectorString vAllString;

    string strReg = R"(\x3F|\x21|\x2C|\x2E|\xC2\xB7|\xEF\xBC\x8C|\xEF\xBC\x9F|\xEF\xBC\x81|\xE3\x80\x82)";
  
    auto  special = regex(strReg);

    for (auto& slist : textList)
    {
        auto strText= ToUTF8(slist);
        sregex_token_iterator it(strText.begin(), strText.end(), special, {-1,1});
        sregex_token_iterator end;

        for (auto seq = it; seq != end; seq++)
        {
            string s = *seq;
            if (s.empty())
            {
                continue;
            }
            //cout << ToGBK(s) << endl;
            vAllString.push_back(s);
        }
          
    }
     
    for (auto& slist : vAllString)
    {
        auto strutf8 =  slist;

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

pair<VectorUint8,int64_t> BBPE::FindMaxPairCount(MapVocabPairCount& vPairCount)
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

void BBPE::CountPairWord(Vector3Uint8& vAllWordList, MapVocabPairCount& vPairCount)
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

            //auto str = VectorUint8ToGBK(merged);
           // cout << str << " : " << vPairCount[merged] << endl;
        }         
    }
        
    for (auto it = vPairCount.begin(); it != vPairCount.end(); )
    {
        if (it->second == 0) 
        {
            it = vPairCount.erase(it);
        }
        else
        {
            ++it;
        }
    }

}

void BBPE::MergeMaxPairWord(Vector3Uint8& vAllWordList, MapVocabPairCount& historyMerge, VectorUint8& tgtKey, int64_t count)
{
    Vector3Uint8 temp;

    for (int i = 0; i < vAllWordList.size(); i++)
    {
        auto& items = vAllWordList[i];
        Vector2Uint8 tempItem;
        for (int j = 0; j < items.size(); j++)
        {
            VectorUint8 merged;

            if (j + 1 < items.size())
            {
                MergeWord(merged, items[j], items[j + 1]);
            }
           
            if (merged == tgtKey)
            {
                tempItem.push_back(merged);
                j += 1;
                historyMerge.emplace(merged, count);
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

void BBPE::SaveFile(const string& path)
{
    ofstream f(path, ios::binary);

    size_t len = m_mapVocabTable.size();

    f.write((const char*) & len, sizeof(len));

    for (auto& item: m_mapVocabTable)
    {
        auto& k= item.first;
        auto& v = item.second;
        len = k.size();
        f.write((const char*)&len, sizeof(len));
        f.write((const char*)k.data(), sizeof(uint8_t) * len);
        f.write((const char*)&v, sizeof(v));
        m_nMaxKey = max(m_nMaxKey, len);
    }

    f.write((const char*)&m_nMaxKey, sizeof(m_nMaxKey));

    
}
bool BBPE::LoadFile(const string& path)
{
    bool b = false;
   
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
    {
        return b;
    }

    size_t count = 0;

    m_mapVocabTable.clear();
    m_mapIDtoCodeId.clear();

    ifs.read((char*)&count, sizeof(count));

    for (int i = 0; i < count; i++)
    {
        VectorUint8 key;
        INT64 v = 0;

        size_t len = 0;
        ifs.read((char*)&len, sizeof(len));
 
        key.reserve(len);
        key.resize(len);
        ifs.read((char*)key.data(), len * sizeof(uint8_t));
        ifs.read((char*)&v, sizeof(v));
        m_mapVocabTable.emplace(key, v);
        m_mapIDtoCodeId.emplace(v, key);
        ///AddNewKeyToVocabTable(key);
    }

    m_nMaxKey = 0;

    ifs.read((char*)&m_nMaxKey, sizeof(m_nMaxKey));

    b = true;

    return b;
}

void BBPE::AddSpecialTokens(const VectorString& tokens)
{
    for (auto& slist : tokens)
    {
        auto strutf8 = ToUTF8(slist);
        VectorUint8 item;
        for (int j = 0; j < strutf8.length(); j++)
        {
            item.push_back(strutf8[j]);
        }

        if (0 < item.size())
        {
            AddNewKeyToVocabTable(item);
        }
    }
    
}


void BBPE::Encode(const string& text, VectorCodeID& ids)
{
    ids.clear();

    auto  special = regex(R"(<[^>]*>)");
  
    sregex_token_iterator it(text.begin(), text.end(), special, { -1, 0 });
    sregex_token_iterator end;

   
    for (auto seq = it; seq != end; ++seq)
    {
        string s = *seq;
        if (s.empty())
        {
            continue;
        }
        Vector2Uint8 vEnumWordList;
        auto  utf8 = ToUTF8(s);
        bool bEncode = false;
        if (utf8.length() <= m_nMaxKey)
        {
            if (IsExistVocabTable(utf8))
            {
                auto w =  GetWordEncode(utf8);
                ids.insert(ids.end(), w.begin(), w.end());
            }
            else
            {
                TokenizerVector(utf8, vEnumWordList);
            }
        }
        else
        {
            TokenizerVector(utf8, vEnumWordList);
        }

        for (int  i = 0; i < vEnumWordList.size(); i++)
        {
            bool b = false;
            VectorUint8 merged = vEnumWordList[i];
            do
            {
                VectorUint8 out;

                if (i + 1 < vEnumWordList.size())
                {
                    MergeWord(out, merged, vEnumWordList[i + 1]);
                }
                else
                {
                    out = merged;
                }
                   
                b = IsExistVocabTable(out);
                if (!b)
                {
                    break;
                }
                merged = out;
                i += 1;

            } while (i + 1 < vEnumWordList.size());

            auto w = GetWordEncode(merged);
            ids.insert(ids.end(), w.begin(), w.end());
            //string strk(merged.begin(), merged.end());
            ///cout << ToGBK(strk)<< endl;
        }
            
    }

    //string strk(ids.begin(), ids.end());
   // cout << Decode(ids) << endl;

}


string BBPE::Decode(const VectorCodeID& ids)
{
    string str="";
    VectorUint8 strList;

    for (auto& k : ids)
    {
        if (m_mapIDtoCodeId.find(k) !=  m_mapIDtoCodeId.end())
        {
            auto vlist = m_mapIDtoCodeId.at(k);
            strList.insert(strList.end(), vlist.begin(), vlist.end());
            //str += VectorUint8ToGBK(vlist);
        }
    }
    str = VectorUint8ToGBK(strList);
    return str;
}

VectorCodeID  BBPE::GetWordEncode(const string& str)
{
    VectorUint8 word;
    for (auto& i:str)
    {
        word.push_back(i);
    }

    return GetWordEncode(word);
}

VectorCodeID BBPE::GetWordEncode(VectorUint8& word)
{
    VectorCodeID res;

    if (m_mapVocabTable.find(word) != m_mapVocabTable.end())
    {
        res.push_back(m_mapVocabTable.at(word));
    }
    else
    {
        for (auto& w: word)
        {
            res.push_back(m_mapVocabTable.at({ w }));
        }
    }

    return res;
}

void BBPE::TokenizerVector(string& textLis, Vector2Uint8& vEnumWordList)
{
    vEnumWordList.clear();
    auto strutf8 = textLis;

    for (int i = 0; i < strutf8.size(); i++)
    {
        int len = GetWordSzie(strutf8[i]);
        VectorUint8 word;

        for (int j = 0; j < len; j++)
        {
            word.push_back(strutf8[i + j]);
        }
        vEnumWordList.push_back(word);
        i += len - 1;
    }
}


int64_t BBPE::GetBOS()
{
   return  GetSpecialString(BOS);

}
int64_t BBPE::GetEOS()
{
    return  GetSpecialString(EOS);
}
int64_t BBPE::GetPAD()
{
    return GetSpecialString(PAD);
}

INT64 BBPE::GetSpecialString(const string& str)
{
    auto strutf8 = ToUTF8(str);
    VectorUint8 item;
    for (int j = 0; j < strutf8.length(); j++)
    {
        item.push_back(strutf8[j]);
    }
    return  m_mapVocabTable.at(item);
}

