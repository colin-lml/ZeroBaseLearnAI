#pragma once

struct XBatchSampler : public torch::data::samplers::RandomSampler
{
	XBatchSampler(size_t size);

	std::optional<std::vector<size_t>> next(size_t batch_size) override;
};



class XBDataset :public torch::data::Dataset<XBDataset>
{
public:
	XBDataset();
	~XBDataset();
	torch::optional<size_t> size() const;
	torch::data::Example<torch::Tensor, torch::Tensor>  get(size_t index) override;
	

	int64_t GetDictionarySize()
	{
		return m_bbpe.GetDictionaryCount();

		
	}
	void  Encode(const string& text, VectorInt64& ids)
	{
		m_bbpe.Encode(text, ids);
	}

	string Decoded(VectorInt64& ids)
	{

		ids.erase(std::remove_if(ids.begin(), ids.end(), [&](const int64_t& id)
			{
				return id == m_iPad || id == m_iBos|| id == m_iEos;
			}), ids.end());

		string str = m_bbpe.Decoded(ids);

		return str;
	}

	int64_t GetBOS()
	{
		return m_iBos;
	}

	int64_t GetEOS()
	{
		return m_iEos;
	}

	int64_t GetPAD()
	{
		return m_iPad;
	}

private:
	XBBPE m_bbpe;
	VectorTrainEncoded& m_vData;
	int64_t m_iBos;
	int64_t m_iEos;
	int64_t m_iPad;
	
};

