#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <regex>
#include <cstdint>
#include <cstring>
#include <windows.h>

using namespace std;


#define VocabSize   600


struct VectorUint8Key
{
    size_t operator()(const vector<uint8_t>& v) const 
    {
        size_t hashKey = 0;
        for (uint8_t b : v) 
        {
            hashKey ^= hash<uint8_t>{}(b)+0x9e3779b9 + (hashKey << 6) + (hashKey >> 2);
        }
        return hashKey;
    }
};

typedef vector<string> VectorString;
typedef vector<uint8_t> VectorUint8;

typedef vector<VectorUint8> VectorVectorUint;

typedef unordered_map<VectorUint8, int64_t, VectorUint8Key> MapVocabTable; /// codeid - > id

typedef unordered_map<int64_t, VectorUint8> MapIDToCodeId; /// id - > codeid

class BBPE
{
public:
    BBPE();
    ~BBPE();
    void Train(const VectorString& textList, uint32_t vocabSize = VocabSize);

//private:
    void InitData();
    int GetWordSzie(uint8_t ch);
    void EnumerationWord(const VectorString& textList, VectorVectorUint& vEnumWordList);

    void CountPairWord(VectorVectorUint& vAllWordList, MapVocabTable& vPairCount);

    pair<VectorUint8, int64_t> FindMaxPairCount(MapVocabTable& vPairCount);

    bool IsExistVocabTable(VectorUint8& v);
    string  ToUTF8(const string& str);
    string  ToGBK(const string& str);
    void AddNewKeyToVocabTable(VectorUint8& vlist);
    string MultiByteToMultiByte(const string& str, UINT from = CP_ACP, UINT bto = CP_UTF8);

    MapVocabTable m_mapVocabTable;
    MapIDToCodeId m_mapIDtoCodeId;
};

