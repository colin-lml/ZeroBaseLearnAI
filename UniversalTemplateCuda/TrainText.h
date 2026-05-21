#pragma once

struct TrainText
{
	string type;
	string title;
	string content;
};

struct TrainEncoded
{
	vector<int64_t> type;
	vector<int64_t> title;
	vector<int64_t> content;

	void write(ofstream& ofs);
	void read(ifstream& ifs);
};


typedef vector<TrainText>  VectorTrainText;
typedef vector<TrainEncoded>  VectorTrainEncoded;