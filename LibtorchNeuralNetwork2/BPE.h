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

using namespace std;

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

class BPE
{
public:
	

};

