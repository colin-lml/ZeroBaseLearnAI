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
