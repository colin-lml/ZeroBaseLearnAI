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
		return m_bbpe.GetCorpusVocabCount();
		
	}
	int64_t GetBOS()
	{
		return m_bbpe.GetBOS();
	}
	int64_t GetEOS()
	{
		return m_bbpe.GetEOS();
	}
	int64_t GetPAD()
	{
		return m_bbpe.GetPAD();
	}

	string Decode(const VectorCodeID& ids)
	{
		return m_bbpe.Decode(ids);
	}

	std::vector<int64_t> Encode(const string& text)
	{
		VectorCodeID ids;

		m_bbpe.Encode(text, ids);

		return ids;
	}

private:
	void LoadDataTxtFile();

	void InitTokenizer(std::vector<Tangshi>& vDataList);

	void saveMap(std::vector<VectorCodeTangshi>& vData);
	bool loadMap();
	void InitEncodeTangshi(std::vector<Tangshi>& vDataList);

	std::vector<Tangshi> m_vdata;
	std::vector<VectorCodeTangshi> m_vEncodeDataList;
	

	const std::string  m_strBinFile = "TokenizerData.bin";
	BBPE m_bbpe;
};

