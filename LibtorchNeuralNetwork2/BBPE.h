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
#define BBPE_PATH   "BBPE_Model.bin"

#define BOS   "<BOS>"
#define EOS   "</BOS>"  
#define PAD   "<PAD>"


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

typedef vector<VectorUint8> Vector2Uint8;

typedef vector<Vector2Uint8> Vector3Uint8;


typedef unordered_map<VectorUint8, int64_t, VectorUint8Key> MapVocabTable; /// codeid - > id

typedef unordered_map<int64_t, VectorUint8> MapIDToCodeId; /// id - > codeid

typedef map<int, string> MapSplitString;

typedef map<VectorUint8, int64_t> MapVocabPairCount;

typedef vector<int64_t> VectorCodeID;


class BBPE
{
public:
    BBPE();
    ~BBPE();
    void Train(const VectorString& textList, uint32_t vocabSize = VocabSize);

    string Decode(const VectorCodeID& ids);

    void Encode(const string& text, VectorCodeID& ids);

private:
    void InitData();
    int  GetWordSzie(uint8_t ch);
    
    void DataCleansingWord(const VectorString& textList, Vector3Uint8& vEnumWordList);

    void CountPairWord(Vector3Uint8& vAllWordList, MapVocabPairCount& vPairCount);
    void MergeMaxPairWord(Vector3Uint8& vAllWordList, MapVocabPairCount& historyMerge, VectorUint8& tgtKey, int64_t count);

    void MergeWord(VectorUint8& outMerge, const VectorUint8& a, const VectorUint8& b);

    pair<VectorUint8, int64_t> FindMaxPairCount(MapVocabPairCount& vPairCount);

    bool IsExistVocabTable(const VectorUint8& v);
    string  ToUTF8(const string& str);
    string  ToGBK(const string& str);
    string  VectorUint8ToGBK(const VectorUint8& item);
    VectorString SplitText(string str, VectorString& delimiter, bool bSave=false);
    void AddNewKeyToVocabTable(const VectorUint8& vlist);
    string MultiByteToMultiByte(const string& str, UINT from = CP_ACP, UINT bto = CP_UTF8);

    void AddSpecialTokens(const VectorString& tokens);

    void SaveFile(const string& path = BBPE_PATH);
    bool LoadFile(const string& path = BBPE_PATH);

    VectorUint8 GetWordEncode(VectorUint8& word);
    void TokenizerVector(string& textLis, Vector2Uint8& vEnumWordList);

    MapVocabTable m_mapVocabTable;
    MapIDToCodeId m_mapIDtoCodeId;

    VectorString m_delimiters;
    size_t m_nMaxKey = 0;
};

