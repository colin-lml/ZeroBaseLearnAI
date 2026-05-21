#include "pch.h"
#include "LogStream.h"

LogStream::LogStream(const string& path, ios_base::openmode type)
{
	auto logPath = GetOutputPath() + path;
    m_ofs.open(logPath.c_str(), type);
}

string LogStream::GetLogTime()
{

    auto now = chrono::system_clock::now();
    std::time_t now_time = chrono::system_clock::to_time_t(now);

    std::tm local_tm{};
    localtime_s(&local_tm, &now_time);

    char buf[64];
    snprintf(buf, sizeof(buf),
        "%04d-%02d-%02d %02d:%02d:%02d  ",
        local_tm.tm_year + 1900,
        local_tm.tm_mon + 1,
        local_tm.tm_mday,
        local_tm.tm_hour,
        local_tm.tm_min,
        local_tm.tm_sec);

    return string(buf);
}



LogStream& LogStream::operator<<(std::ostream& (*manip)(std::ostream&))
{
    std::lock_guard<std::mutex> lock(m_mutex);

    {
        m_ss << "\n";
        std::string content = m_ss.str();

        std::cout << content;
        
        if (m_ofs.is_open())
        {
            m_ofs << content;
            m_ofs.flush();
        }
     
        m_ss.str("");
        m_ss.clear();
    }
    return *this;
}