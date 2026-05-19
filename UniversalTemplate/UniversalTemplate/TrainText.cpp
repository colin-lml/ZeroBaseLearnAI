#include "pch.h"
#include "TrainText.h"



void TrainEncoded::write(ofstream& ofs)
{
    size_t len = type.size();
    ofs.write((const char*)&len, sizeof(len));
    ofs.write((const char*)type.data(), len * sizeof(int64_t));

    len = title.size();
    ofs.write((const char*)&len, sizeof(len));
    ofs.write((const char*)title.data(), len * sizeof(int64_t));

    len = content.size();
    ofs.write((const char*)&len, sizeof(len));
    ofs.write((const char*)content.data(), len * sizeof(int64_t));
}

void TrainEncoded::read(ifstream& ifs)
{
    size_t len = 0;
    ifs.read((char*)&len, sizeof(len));
    type.resize(len);
    ifs.read((char*)type.data(), len * sizeof(int64_t));

    ifs.read((char*)&len, sizeof(len));
    title.resize(len);
    ifs.read((char*)title.data(), len * sizeof(int64_t));

    ifs.read((char*)&len, sizeof(len));
    content.resize(len);
    ifs.read((char*)content.data(), len * sizeof(int64_t));

}