#include "pch.h"
#include "XBBPE.h"



string GetOutputPath()
{
    auto s = std::filesystem::current_path().string();
    size_t len = s.size() - s.find_last_of("\\");
   
   
    s.erase(s.end() - len, s.end());
    
    s += "\\tmpbin\\";

    if (!filesystem::exists(s))
    {
        filesystem::create_directories(s);
    }

    return s;
}


XBBPE::XBBPE()
{
   
    /* 
    VectorString corpus =
    {
        "用电电电鳗电鳗会不会被电电死?",
        "bbpe 是 byte level bpe 分词算法。",
        "bpe 算法用于大模型 token 编码。",
        "bbpe 基于 utf8 字节合并中文英文。",
        "token 编码电鳗放电测试。",
        "token  to a ab abc abc  abcd abcf.,，。"
    };
    */
    

    if (!LoadFile())
    {
        InitData();

        LoadDataFileTrain("tangshi.data.txt");
        //  LoadDataFileTrain("HelloWorld.txt");
        
    }

     
    for (auto& item: m_vectorTrainEncoded)
    {
        string a =  Decoded(item);
        cout << a << endl;
    }
    
}


XBBPE::~XBBPE()
{

}



void XBBPE::LoadDataFileTrain(const string& paths, uint32_t vocabSize)
{
    m_vectorTrainText.clear();

    auto xPath = GetOutputPath() + paths;
    std::ifstream ifs(xPath);
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

        TrainText item;

        item.type = line + "\n";

        if (getline(ss, line))
        {
            item.title = line + "\n";
        }

        while (getline(ss, line))
        {
            if (line.empty())
            {
                break;
            }
            item.content += line + "\n";
        }
        m_vectorTrainText.push_back(item);
    }


    VectorString vstring;

    for(auto& v : m_vectorTrainText)
    {
        vstring.push_back(v.type);
        vstring.push_back(v.title);
        vstring.push_back(v.content);
    }

    Train(vstring, vocabSize);

    for (auto& v : m_vectorTrainText)
    {
        TrainEncoded item;
        Encode(v.type, item.type);
        Encode(v.title, item.title);
        Encode(v.content, item.content);

        m_vectorTrainEncoded.push_back(item.GetAllData());
    }

    SaveFile();
}


bool XBBPE::LoadFile(const string& path)
{
    auto binPath =  GetOutputPath() + path;
    ifstream  infs(binPath, ios::binary);
    if (!infs.is_open())
    {
        return false;
    }

    m_vectorTrainEncoded.clear();
    m_mapEncoderList.clear();
    m_mapDecoderList.clear();
    size_t count = 0;
    infs.read((char*)&count, sizeof(size_t));

    for (size_t i = 0; i < count; i++)
    {
        pair<WordIdKey, int64_t> item;
        infs.read((char*)&item, sizeof(item));
        m_mapEncoderList.emplace(item);
        m_mapDecoderList.emplace(item.second, item.first);
    }

    count = 0;
    infs.read((char*)&count, sizeof(size_t));

    for (int i = 0; i < count; i++)
    {
        size_t len = 0;
        infs.read((char*)&len, sizeof(size_t));
        VectorInt64 item;
        item.resize(len);
        infs.read((char*)item.data(), len * sizeof(int64_t));
        m_vectorTrainEncoded.push_back(item);
    }

    infs.close();

    return !m_mapEncoderList.empty();
}

void XBBPE::SaveFile(const string& path)
{
    auto binPath = GetOutputPath() + path;

    remove(binPath.c_str());

    ofstream  outfs(binPath, ios::binary);

    size_t count = m_mapEncoderList.size();

    outfs.write((const char* ) & count, sizeof(count));

    for (const auto& pair : m_mapEncoderList)
    {
        outfs.write((const char*)&pair, sizeof(pair));
    }


    count = m_vectorTrainEncoded.size();
    outfs.write((const char*)&count, sizeof(count));

    for (auto& v : m_vectorTrainEncoded)
    {
        count = v.size();
        outfs.write((const char*)&count, sizeof(count));
        outfs.write((const char*)v.data(), count * sizeof(int64_t));
    }

    outfs.close();

}

void XBBPE::InitData(void)
{
    m_mapEncoderList.clear();
    m_mapDecoderList.clear();

    for (int i = 0; i < 256; i++)
    {
        VectorUint8 b;
        b.push_back(i);
        AddNewKeyToWordList(b);
    }

    std::vector<VectorUint8> filterSyms =
    {
        {0xC2, 0xB7},
        {0xEF, 0xBC, 0x8C},
        {0xEF, 0xBC, 0x9F},
        {0xEF, 0xBC, 0x81},
        {0xE3, 0x80, 0x82},
        {0xE3, 0x80, 0x80},
        {0xE2, 0x80, 0x8B}
    };

    for (auto& f : filterSyms)
    {
        AddNewKeyToWordList(f);
    }
}

int  XBBPE::GetWordSize(uint8_t ch)
{
    
    int len = 1;
    if ((ch & 0x80) == 0)
    {
        len = 1; // ASCII
    }
    else if ((ch & 0xE0) == 0xC0)
    {
        len = 2;
    }
    else if ((ch & 0xF0) == 0xE0)
    {
        len = 3; // 中文
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

bool XBBPE::IsInWordList(const WordIdKey& key)
{
    return m_mapEncoderList.find(key) != m_mapEncoderList.end();
}

void XBBPE::AddNewKeyToWordList(const VectorUint8& vKey)
{
    string key(vKey.begin(), vKey.end());
    AddNewKeyToWordList({ key });
}

void XBBPE::AddNewKeyToWordList(const WordIdKey& key)
{
    auto id = m_mapEncoderList.size();

    if (!IsInWordList(key))
    {
        m_mapEncoderList.emplace(key, id);
        m_mapDecoderList.emplace(id, key);
    }

}


string  XBBPE::ToUTF8(const string& strGbk)
{
    return MultiByteToMultiByte(strGbk, CP_ACP, CP_UTF8);
}

string  XBBPE::ToGBK(const string& strUtf8)
{
    return MultiByteToMultiByte(strUtf8, CP_UTF8, CP_ACP);
}


string XBBPE::MultiByteToMultiByte(const string& str, UINT from, UINT bto)
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


void XBBPE::Train(const VectorString& textList, uint32_t vocabSize)
{

    vocabSize = vocabSize - 7;

    string strReg = R"(\x0A|\x3F|\x20|\x21|\x22|\x2C|\x2E|\xC2\xB7|\xEF\xBC\x8C|\xEF\xBC\x9F|\xEF\xBC\x81|\xE3\x80\x82|\xE3\x80\x80|\xE2\x80\x8B)";
    auto  special = regex(strReg);

    InitData();

    WordIdKey key;
    Vector2Word v2WordList;

    for (auto& slist : textList)
    {
        auto strText = ToUTF8(slist);
        sregex_token_iterator it(strText.begin(), strText.end(), special, { -1,1 });
        sregex_token_iterator end;

        for (auto seq = it; seq != end; seq++)
        {
            VectorWord vItem;

            string s = *seq;
            if (s.empty())
            {
                continue;
            }
            //cout << ToGBK(s) << endl;
            for (int i = 0; i < s.size(); i++)
            {
                int len = GetWordSize(s[i]);
                VectorUint8 word;
                for (int j = 0; j < len; j++)
                {
                    word.push_back(s[i + j]);
                }
                i += len - 1;
                vItem.push_back({ word });
            }

            if (1 < vItem.size())
            {
                v2WordList.push_back(vItem);
            }
        }
 
    }
   

    VectorWord vSingleWordList;
    bool del = true;
    while (m_mapEncoderList.size() < vocabSize)
    {
        WordIdKey addKey =  MergeMaxPairWord(v2WordList, vSingleWordList,del);

        if (del)
        {
            for (auto& vd : vSingleWordList)
            {
                if (m_mapEncoderList.size() < vocabSize)
                {
                    AddNewKeyToWordList(vd);
                }
                else
                {
                    break;
                }
            }
        }

        del = false;

       if (addKey.len == 0 || vocabSize <= m_mapEncoderList.size()  )
       {
           break;
       }

       //string kk((char*)addKey.idKey);
       //cout << ToGBK(kk) << endl;
       AddNewKeyToWordList(addKey);
    }

   

    VectorString sp;
    sp.push_back(PAD);
    sp.push_back(BOS);
    sp.push_back(EOS);

    AddSpecialTokens(sp);

}

void XBBPE::AddSpecialTokens(const VectorString& tokens)
{
    for (auto& slist : tokens)
    {
        auto strutf8 = ToUTF8(slist);
             
        for (int i=0; i < strutf8.size(); i++)
        {
            VectorUint8 item(strutf8.begin(), strutf8.begin()+1+i);
            AddNewKeyToWordList(item);
        }

    }

}


WordIdKey& XBBPE::MergeMaxPairWord(Vector2Word& v2WordList, VectorWord& vSingleWordList, bool del)
{
    MapEncoderWordList single;
    MapEncoderWordList merge;
    VectorWord  maxlist;
    size_t maxPair = 0;
    WordIdKey maxWord;
    

    for (auto& list : v2WordList)
    {
        for (size_t i = 0; i+1 < list.size(); i++)
        {
            WordIdKey m;
            m.Append(list[i]);
            m.Append(list[i+1]);

            if (merge.find(m) == merge.end())
            {
                merge.emplace(m, 0);
            }
            else
            {
                merge[m]++;
            }

            if (maxPair < merge[m])
            {
                maxPair = merge[m];
                maxWord = m;
               
            }
        }
    }


    for (auto& list : v2WordList)
    {
        for (int i = 0; i + 1 < list.size(); i++)
        {
            WordIdKey m;
            if (1 <= i)
            {
                m.Append(list[i - 1]);
                m.Append(list[i]);
            }

            WordIdKey m2;
            m2.Append(list[i]);
            m2.Append(list[i + 1]);

            if (merge[m] == 0 && merge[m2] == 0 && del)
            {
                if (single.find(list[i]) == single.end())
                {
                    single.emplace(list[i], 0);
                }
                else
                {
                    single[list[i]]++;
                }
                
                list.erase(list.begin()+i);
                i--;
            }
            else if (maxWord == m2 && merge[m2] == maxPair)
            {
                list[i] = maxWord;
                list.erase(list.begin() + i + 1);
            }
        }
          
    }
  
    if (del)
    {
        for (auto& key : single)
        {
            if (0 < key.second && 3 <= key.first.len)
            {
                vSingleWordList.push_back(key.first);
            }
        }
        
        v2WordList.erase(std::remove_if(v2WordList.begin(), v2WordList.end(), [&](const VectorWord& vw)
            {
                bool b = vw.size() <= 1;
                return b;
            }), v2WordList.end());
    }


    return maxWord;
}

int64_t XBBPE::GetBOS()
{
    int64_t id = 0;
    VectorInt64 ids;
    Encode(BOS, ids);
    if (0 < ids.size())
    {
        id = ids.at(0);
    }

    return id;
}
int64_t XBBPE::GetEOS()
{
    int64_t id = 0;
    VectorInt64 ids;
    Encode(EOS, ids);
    if (0 < ids.size())
    {
        id = ids.at(0);
    }

    return id;
}
int64_t XBBPE::GetPAD()
{
    int64_t id = 0;
    VectorInt64 ids;
    Encode(PAD, ids);
    if (0 < ids.size())
    {
        id = ids.at(0);
    }

    return id;
}


void XBBPE::Encode(const string& textGbk, VectorInt64& ids)
{
    ids.clear();
    auto  special = regex(R"(<[^>]*>)");
    auto text = ToUTF8(textGbk);
    sregex_token_iterator it(text.begin(), text.end(), special, { -1, 0 });
    sregex_token_iterator end;

    for (auto seq = it; seq != end; ++seq)
    {
        string s = *seq;
        if (s.empty())
        {
            continue;
        }
        VectorWord vWordList;
        ToTextVectorWord(s, vWordList);

        for (size_t i = 0; i < vWordList.size(); i++)
        {

            WordIdKey m(vWordList[i]);

            do
            {
                WordIdKey m2 = m;
                if (i+1 < vWordList.size())
                {
                    m2.Append(vWordList[i + 1]);
                }

                if (!IsInWordList(m2))
                {
                    break;
                }
                m = m2;
                i += 1;

            } while (i + 1 < vWordList.size());

            GetWordEncode(m, ids);
          
        }
        
    }

}

void XBBPE::ToTextVectorWord(const string& strUtf8, VectorWord& vWordList)
{
    vWordList.clear();

    for (size_t i = 0; i < strUtf8.size(); i++)
    {
        int len = GetWordSize(strUtf8[i]);
        VectorUint8 word;
        for (int j = 0; j < len; j++)
        {
            word.push_back(strUtf8[i + j]);
        }
        i += len - 1;
        vWordList.push_back({ word });
    }
}

void XBBPE::GetWordEncode(WordIdKey& word, VectorInt64& vList)
{
    if (m_mapEncoderList.find(word) != m_mapEncoderList.end())
    {
        vList.push_back(m_mapEncoderList.at(word));
    }
    else
    {
        for (int i = 0; i < word.len; i++)
        {
            WordIdKey tm;
            tm.idKey[0] = word.idKey[i];
            tm.len = 1;
            vList.push_back(m_mapEncoderList.at(tm));
        }     
    }
}

string XBBPE::Decoded(const VectorInt64& ids)
{
    
    VectorUint8 vList;
    for (auto& id : ids)
    {
       auto word = m_mapDecoderList.at(id);

       for (int i = 0;i < word.len; i++)
       {
           vList.push_back(word.idKey[i]);
       }
    }

    string str(vList.begin(), vList.end());
    str = ToGBK(str);
    return str;
}

