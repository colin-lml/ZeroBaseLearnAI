#pragma once
class LogStream
{
public:
	LogStream(const string& path, ios_base::openmode type);
    string GetLogTime();

    template<typename T>
    LogStream& operator<<(const T& val)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_ss.tellp() == 0)
        {
            m_ss << GetLogTime();
        }

        m_ss << val;
        return *this;
    }

    LogStream& operator<<(std::ostream& (*manip)(std::ostream&));
 
 
private:
	ofstream m_ofs;
	ostringstream m_ss;
    mutex m_mutex;
};

