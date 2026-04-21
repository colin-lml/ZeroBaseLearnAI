#pragma once
#include <fstream>
#include <iostream>
#include "json.hpp"


#define PadId 0

using json = nlohmann::json;

typedef struct _Tangshi
{
	std::string title;
	std::string author;
	std::string content;

}Tangshi;


typedef std::unordered_map<std::string, int64_t> CorpusVocabStoID;
typedef std::unordered_map<int64_t, std::string> CorpusVocabIDtoS;

class Tokenizer
{
public:
	void InitLoadDataSrc();
	std::vector<int64_t> GetTangshiCode(std::string& line);
	std::string GetTangshiString(std::vector<int64_t>& vList);

	std::vector<std::vector<int64_t>>& GetEncodeData()
	{
		return m_vEncodeDataList;
	}

	int64_t GetCorpusVocabCount()
	{
		return m_stringToID.size();
	}

private:
	void LoadDataTxtFile();
	int ChineseCount(const std::string& s);
	bool IsChinese(const char& ch);
	std::vector<std::string> SplitString(std::string line);
	void InitTokenizer(std::vector<Tangshi>& vDataList);

	void AddVocabTable(std::vector<std::string>& stringList,CorpusVocabStoID&stringID, CorpusVocabIDtoS& IDString);

	void saveMap(CorpusVocabStoID& map1, std::vector<std::vector<int64_t>>& vData);
	bool loadMap();
	void InitEncodeTangshi(std::vector<Tangshi>& vDataList);

	std::vector<Tangshi> m_vdata;
	std::vector<std::vector<int64_t>> m_vEncodeDataList;

	int m_nMaxTitle = 0;
	int m_nMaxAuthor = 0;
	int m_nMaxContent = 0;

	CorpusVocabStoID m_stringToID;
	CorpusVocabIDtoS m_IDToString;
	
	const std::string  m_strBinFile = "TokenizerData.bin";
};

