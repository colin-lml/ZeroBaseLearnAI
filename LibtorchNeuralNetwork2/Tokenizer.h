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
	void LoadDataSrc();

private:
	void LoadDataTxtFile();
	int ChineseCount(const std::string& s);
	bool IsChinese(const char& ch);
	std::vector<std::string> SplitString(std::string line);
	void InitTokenizer(std::vector<Tangshi>& vDataList);

	void AddVocabTable(std::vector<std::string>& stringList,CorpusVocabStoID&stringID, CorpusVocabIDtoS& IDString);

	void saveMap(CorpusVocabStoID& map1);
	bool loadMap();

	std::vector<Tangshi> m_vdata;

	int m_nMaxTitle = 0;
	int m_nMaxAuthor = 0;
	int m_nMaxContent = 0;

	CorpusVocabStoID m_stringToID;
	CorpusVocabIDtoS m_IDToString;
	
	const std::string  m_strBinFile = "TokenizerData.bin";
};

