#pragma once
#define VocabSize   600

#define BBPE_PATH   "xBBPE.bin"
#define MaxKeyCount  (3*6)


typedef vector<string> VectorString;
typedef vector<uint8_t> VectorUint8;

struct WordIdKey
{
	WordIdKey()
	{

	}

	WordIdKey(const VectorUint8& key)
	{
		len = key.size();
		memcpy(idKey, key.data(), min(key.size(), sizeof(idKey)));
	}

	bool Append(const WordIdKey& k)
	{
		bool b = false;
		if (len + k.len < sizeof(idKey))
		{
			memcpy(idKey + len, k.idKey,k.len);
			len += k.len;
			b = true;
		}

		return b;
	}

	WordIdKey(const string& key)
	{
		len = key.size();
		memcpy(idKey, key.data(), min(key.size(), sizeof(idKey)));
		
	}

	uint8_t idKey[MaxKeyCount] ={ 0 };
	size_t len = 0;

	bool operator == (const WordIdKey& other) const
	{
		return len == other.len && memcmp(idKey, other.idKey, sizeof(idKey)) == 0;
	}

	bool operator < (const WordIdKey& other) const
	{
		if (len != other.len)
		{
			return len < other.len;
		}
			
		return memcmp(idKey, other.idKey, sizeof(idKey)) < 0;
	}
	void operator = (const WordIdKey& other)
	{
		if (this == &other)
		{
			return ;
		}

		len = other.len;
		memcpy(idKey, other.idKey, sizeof(idKey));

	}
};

template<>
struct std::hash<WordIdKey>
{
	size_t operator()(const WordIdKey& k) const noexcept
	{
		size_t h = 0;
		for (int i = 0; i < k.len; ++i) 
		{
			h ^= std::hash<uint8_t>{}(k.idKey[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
		}
		return h;
	}
};


typedef vector<WordIdKey> VectorWord;
typedef vector<VectorWord> Vector2Word;

typedef unordered_map<WordIdKey, int64_t> MapEncoderWordList; /// 
typedef unordered_map<int64_t, WordIdKey> MapDecoderWordList; /// 

//typedef map<WordIdKey, int64_t> MapPairWordCount;
//typedef vector<pair<size_t, size_t>> VectorPairWordIndex;


string GetOutputPath();

class XBBPE
{
public:
	XBBPE();
	~XBBPE();

	void LoadDataFileTrain(const string& paths, uint32_t vocabSize = VocabSize);

private:

	void Train(const VectorString& textList, uint32_t vocabSize);

	void InitData(void);
	bool LoadFile(const string& path = BBPE_PATH);
	void SaveFile(const string& path = BBPE_PATH);

	int GetWordSize(uint8_t ch);
	bool IsInWordList(const WordIdKey& key);
	void AddNewKeyToWordList(const VectorUint8& vKey);
	void AddNewKeyToWordList(const WordIdKey& key);

	string  ToGBK(const string& strUtf8);
	string  ToUTF8(const string& strGbk);
	string MultiByteToMultiByte(const string& str, UINT from, UINT bto);

	WordIdKey& MergeMaxPairWord(Vector2Word& v2WordList, VectorWord& vDelWordList, bool del);
	
private:
	MapEncoderWordList m_mapEncoderList;
	MapDecoderWordList m_mapDecoderList;
	VectorTrainText    m_vectorTrainText;
};

