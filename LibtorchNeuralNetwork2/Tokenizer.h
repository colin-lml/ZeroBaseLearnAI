#pragma once
#include <fstream>
#include <iostream>
#include "json.hpp"

#include "BBPE.h"



typedef struct _Tangshi
{
	std::string title;
	std::string author;
	std::string content;

}Tangshi;

struct VectorCodeTangshi
{
	VectorCodeID title;
	VectorCodeID author;
	VectorCodeID content;
};



class Tokenizer
{
public:
	void InitLoadDataSrc();
	
	std::vector<VectorCodeTangshi>& GetEncodeData()
	{
		return m_vEncodeDataList;
	}

	int64_t GetCorpusVocabCount()
	{
		//return m_stringToID.size();
		return 0;
	}

private:
	void LoadDataTxtFile();

	void InitTokenizer(std::vector<Tangshi>& vDataList);

	void saveMap(std::vector<VectorCodeTangshi>& vData);
	bool loadMap();
	void InitEncodeTangshi(std::vector<Tangshi>& vDataList);

	std::vector<Tangshi> m_vdata;
	std::vector<VectorCodeTangshi> m_vEncodeDataList;
	
	//std::vector<std::vector<int64_t>> m_vEncodeDataList;

	const std::string  m_strBinFile = "TokenizerData.bin";
	BBPE m_bbpe;
};

