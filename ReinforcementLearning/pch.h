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

// 添加要在此处预编译的标头
#include <iostream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <random>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
using namespace std;

#define ROW 4
#define COL 8
#define MaxAction  4

#include "CliffWalkingEnv.h"
#include "PolicyIteration.h"
#include "FreeModel.h"
#include "CartPoleEnv.h"
#include "DeepQNetwork.h"
#include "DuelingDQN.h"
#include "Reinforce.h"

#endif //PCH_H
