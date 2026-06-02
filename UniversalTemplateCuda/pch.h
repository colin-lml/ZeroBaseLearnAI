// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#undef ERROR

#include <torch/torch.h>
#include <iostream>
using namespace std;

#include <windows.h>


#include <random>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <regex>
#include <cstdint>
#include <cstring>
#include <filesystem>


#include "TrainText.h"
#include "XBBPE.h"

#include "XMultiHeadAttention.h"
#include "XFeedforward.h"
#include "EmbeddingWithPosition.h"
#include "XDecoderLayer.h"
#include "DecoderOnly.h"

#include "XBDataset.h"
#include "XTrainPredict.h"
#include "LogStream.h"
#include "XRotaryEmbedding.h"



//D:\libtorch_gpu2.11.0\debug\lib\ * .lib
//PATH = D:\libtorch_gpu2.11.0\debug\lib; % PATH%

// D:\libtorch_gpu2.11.0\debug\include
// D:\libtorch_gpu2.11.0\debug\include\torch\csrc\api\include\
// D:\libtorch_gpu2.11.0\debug\lib




#endif //PCH_H
