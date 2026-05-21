#pragma once

typedef vector<int64_t> VectorInt64;

struct TrainText
{
	string type;
	string title;
	string content;
};

struct TrainEncoded
{
	VectorInt64 type;
	VectorInt64 title;
	VectorInt64 content;

	VectorInt64& GetAllData();

private:
	VectorInt64 dataList;

};


typedef vector<TrainText>  VectorTrainText;
typedef vector<VectorInt64>  VectorTrainEncoded;