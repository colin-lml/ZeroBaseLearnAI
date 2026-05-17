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
   
    VectorString corpus =
    {
        "用电电电鳗电鳗会不会被电电死?",
        "bbpe 是 byte level bpe 分词算法。",
        "bpe 算法用于大模型 token 编码。",
        "bbpe 基于 utf8 字节合并中文英文。",
        "token 编码电鳗放电测试。",
        "token  to a ab abc abc  abcd abcf.,，。"
    };

    Train(corpus);

    if (!LoadFile())
    {
        InitData();

        SaveFile();
    }

}


XBBPE::~XBBPE()
{

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
    string strReg = R"(\x3F|\x21|\x2C|\x2E|\xC2\xB7|\xEF\xBC\x8C|\xEF\xBC\x9F|\xEF\xBC\x81|\xE3\x80\x82)";
    auto  special = regex(strReg);
   
    WordIdKey key;
    Vector2Word vWordList;

    for (auto& slist : textList)
    {
        auto strText = ToUTF8(slist);
        sregex_token_iterator it(strText.begin(), strText.end(), special, { -1,1 });
        sregex_token_iterator end;

        VectorWord vWordItem;

        for (auto seq = it; seq != end; seq++)
        {
            string s = *seq;
            if (s.empty())
            {
                continue;
            }

            for (int i = 0; i < s.size(); i++)
            {
                int len = GetWordSize(s[i]);
                VectorUint8 word;
                for (int j = 0; j < len; j++)
                {
                    word.push_back(s[i + j]);
                }
                i += len - 1;
                vWordItem.push_back({ word });
            }
            ///cout << ToGBK(s) << endl;
            ///vAllString.push_back(s);
        }
        vWordList.push_back(vWordItem);
    }




}

void XBBPE::MergeWord(WordIdKey& outMerge, const WordIdKey& a, const WordIdKey& b)
{
    outMerge.Append(a);
    outMerge.Append(b);
   
}
void XBBPE::CountPairWord(Vector2Word& v2WordList)
{
    for (auto& list : v2WordList)
    {
        for (size_t i = 0; i+1 < list.size(); i++)
        {
            WordIdKey m;
            m.Append(list[i]);
            m.Append(list[i+1]);
             
        }
    }

}
