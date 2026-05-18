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

        SaveFile();
    }

    LoadDataFileTrain("tangshi.data.txt");
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

        item.type = line;

        if (getline(ss, line))
        {
            item.title = line;
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
}


bool XBBPE::LoadFile(const string& path)
{
    auto binPath =  GetOutputPath() + path;
    ifstream  infs(binPath, ios::binary);
    if (!infs.is_open())
    {
        return false;
    }

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
    vocabSize = vocabSize - 3;

    string strReg = R"(\x0A|\x3F|\x20|\x21|\x2C|\x2E|\xC2\xB7|\xEF\xBC\x8C|\xEF\xBC\x9F|\xEF\xBC\x81|\xE3\x80\x82|\xE3\x80\x80|\xE2\x80\x8B)";
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
   
     /* 
    for (auto& ls : v2WordList)
    {
        for (auto& w: ls)
        {
            string kk((char*)w.idKey);
            cout << ToGBK(kk)<<"";
        }

        cout  << endl;
    }
    cout << endl << endl;
    */

    VectorWord vDelWordList;
    bool del = true;
    while (m_mapEncoderList.size() < vocabSize)
    {
        WordIdKey addKey =  MergeMaxPairWord(v2WordList, vDelWordList,del);
        del = false;

       if (addKey.len == 0)
       {
           break;
       }

      // string kk((char*)addKey.idKey);
      // cout << ToGBK(kk) << endl;
       AddNewKeyToWordList(addKey);
    }

    for (auto& vd : vDelWordList)
    {
        if (m_mapEncoderList.size() < vocabSize)
        {
             string kk((char*)vd.idKey);
             cout << ToGBK(kk) << endl;
            AddNewKeyToWordList(vd);
        }
        else
        {
            break;
        }
    }


    SaveFile();

}


WordIdKey& XBBPE::MergeMaxPairWord(Vector2Word& v2WordList, VectorWord& vDelWordList, bool del)
{
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
                vDelWordList.push_back(list[i]);
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
        v2WordList.erase(std::remove_if(v2WordList.begin(), v2WordList.end(), [&](const VectorWord& del)
            {
                bool b = del.size() <= 1;
               
                if (b && del.size() == 1)
                {
                    vDelWordList.push_back(del.at(0));
                }

                return b;
            }), v2WordList.end());
    }


    /* 
    {
       
        for (auto& ls : v2WordList)
        {
            for (auto& w : ls)
            {
                string kk((char*)w.idKey);
                cout << ToGBK(kk) << " ";
            }

            cout << endl;
        }
        
    }
     */

    return maxWord;
}
