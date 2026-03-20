#include "learning/NeuralNet.h"

#include <cmath>
#include <fstream>
#include <cstring>

#include <json/json.h>

cNeuralNet::cNeuralNet()
{
	Clear();
}

void cNeuralNet::Clear()
{
	mPolicyModel = ePolicyModel::eDeepRL;
	mInputSize = 0;
	mOutputSize = 0;
	mInputOffset.resize(0);
	mInputScale.resize(0);
	mOutputOffset.resize(0);
	mOutputScale.resize(0);
	mLayerState.clear();
	mQTable.clear();
#if AP_DTRL_HAS_TORCH
	mHasModule = false;
#endif
}

bool cNeuralNet::HasNet() const
{
#if AP_DTRL_HAS_TORCH
	if (mPolicyModel != ePolicyModel::eClassicRL && mHasModule)
	{
		return true;
	}
#endif
	return mPolicyModel == ePolicyModel::eClassicRL && !mQTable.empty();
}

bool cNeuralNet::HasLayer(const std::string& layer_name) const
{
	return mLayerState.find(layer_name) != mLayerState.end();
}

void cNeuralNet::LoadNet(const std::string& net_file)
{
	std::ifstream f(net_file);
	Json::Value root;
	f >> root;

	const std::string model_kind = root.get("policy_model", "deep_rl").asString();
	mPolicyModel = ParsePolicyModel(model_kind);

	mInputSize = root.get("input_size", 0).asInt();
	mOutputSize = root.get("output_size", 0).asInt();

	const std::string model_file = root.get("model_file", "").asString();
	if (!model_file.empty())
	{
		LoadModel(model_file);
	}

	const std::string scale_file = root.get("scale_file", "").asString();
	if (!scale_file.empty())
	{
		LoadScale(scale_file);
	}
	else
	{
		InitIdentityScales();
	}

	const Json::Value q_table = root["q_table"];
	if (q_table.isArray())
	{
		for (Json::ArrayIndex i = 0; i < q_table.size(); ++i)
		{
			const Json::Value& row = q_table[i];
			if (!row.isObject())
			{
				continue;
			}

			const Json::Value state = row["state"];
			const Json::Value values = row["values"];
			if (!state.isArray() || !values.isArray())
			{
				continue;
			}

			Eigen::VectorXd s(state.size());
			for (Json::ArrayIndex j = 0; j < state.size(); ++j)
			{
				s[j] = state[j].asDouble();
			}

			Eigen::VectorXd v(values.size());
			for (Json::ArrayIndex j = 0; j < values.size(); ++j)
			{
				v[j] = values[j].asDouble();
			}

			if (mOutputSize == 0)
			{
				mOutputSize = static_cast<int>(v.size());
			}
			mQTable[BuildQKey(s)] = v;
		}
	}
}

void cNeuralNet::LoadModel(const std::string& model_file)
{
#if AP_DTRL_HAS_TORCH
	if (mPolicyModel != ePolicyModel::eClassicRL)
	{
		mModule = torch::jit::load(model_file);
		mHasModule = true;
		return;
	}
#endif
	(void)model_file;
}

void cNeuralNet::LoadScale(const std::string& scale_file)
{
	std::ifstream f(scale_file);
	Json::Value root;
	f >> root;

	auto parse_vec = [](const Json::Value& node)
	{
		Eigen::VectorXd v(node.size());
		for (Json::ArrayIndex i = 0; i < node.size(); ++i)
		{
			v[i] = node[i].asDouble();
		}
		return v;
	};

	if (root.isMember("input_offset")) mInputOffset = parse_vec(root["input_offset"]);
	if (root.isMember("input_scale")) mInputScale = parse_vec(root["input_scale"]);
	if (root.isMember("output_offset")) mOutputOffset = parse_vec(root["output_offset"]);
	if (root.isMember("output_scale")) mOutputScale = parse_vec(root["output_scale"]);

	if (mInputSize == 0) mInputSize = static_cast<int>(mInputOffset.size());
	if (mOutputSize == 0) mOutputSize = static_cast<int>(mOutputOffset.size());
	InitIdentityScales();
}

void cNeuralNet::CopyModel(const cNeuralNet& other)
{
	*this = other;
}

void cNeuralNet::OutputModel(const std::string& out_file) const
{
	Json::Value root;
	root["policy_model"] = (mPolicyModel == ePolicyModel::eClassicRL)
		? "classic_rl"
		: (mPolicyModel == ePolicyModel::eDeepRLAttention ? "deep_rl_attention" : "deep_rl");
	root["input_size"] = mInputSize;
	root["output_size"] = mOutputSize;

	std::ofstream f(out_file);
	f << root;
}

void cNeuralNet::Eval(const Eigen::VectorXd& x, Eigen::VectorXd& out_y) const
{
	if (mPolicyModel == ePolicyModel::eClassicRL)
	{
		out_y = RunClassic(x);
		return;
	}

	if (mPolicyModel == ePolicyModel::eDeepRLAttention)
	{
		out_y = RunTorch(ApplyAttention(x));
		return;
	}

	out_y = RunTorch(x);
}

void cNeuralNet::ForwardInjectNoisePrefilled(const Eigen::VectorXd& noise_mean,
									 const Eigen::VectorXd& noise_stdev,
									 const std::string& layer_name,
									 Eigen::VectorXd& out_y) const
{
	Eigen::VectorXd x;
	GetLayerState(layer_name, x);
	if (x.size() == 0)
	{
		x = Eigen::VectorXd::Zero(mInputSize);
	}

	Eigen::VectorXd noisy = x;
	for (int i = 0; i < noisy.size(); ++i)
	{
		const double mean = (i < noise_mean.size()) ? noise_mean[i] : 0.0;
		const double stdev = (i < noise_stdev.size()) ? noise_stdev[i] : 0.0;
		noisy[i] += mean + cMathUtil::RandDoubleNorm(0.0, stdev);
	}

	out_y = RunTorch(noisy);
}

void cNeuralNet::NormalizeInput(Eigen::VectorXd& inout_x) const
{
	if (inout_x.size() == mInputOffset.size())
	{
		inout_x += mInputOffset;
	}
	if (inout_x.size() == mInputScale.size())
	{
		inout_x = inout_x.cwiseProduct(mInputScale);
	}
}

void cNeuralNet::NormalizeOutput(Eigen::VectorXd& inout_y) const
{
	if (inout_y.size() == mOutputOffset.size())
	{
		inout_y += mOutputOffset;
	}
	if (inout_y.size() == mOutputScale.size())
	{
		inout_y = inout_y.cwiseProduct(mOutputScale);
	}
}

int cNeuralNet::GetInputSize() const { return mInputSize; }
int cNeuralNet::GetOutputSize() const { return mOutputSize; }
const Eigen::VectorXd& cNeuralNet::GetOutputOffset() const { return mOutputOffset; }
const Eigen::VectorXd& cNeuralNet::GetOutputScale() const { return mOutputScale; }

void cNeuralNet::SetLayerState(const Eigen::VectorXd& state, const std::string& layer_name)
{
	mLayerState[layer_name] = state;
}

void cNeuralNet::GetLayerState(const std::string& layer_name, Eigen::VectorXd& out_state) const
{
	auto it = mLayerState.find(layer_name);
	if (it != mLayerState.end())
	{
		out_state = it->second;
	}
	else
	{
		out_state = Eigen::VectorXd();
	}
}

cNeuralNet::ePolicyModel cNeuralNet::GetPolicyModel() const { return mPolicyModel; }
void cNeuralNet::SetPolicyModel(ePolicyModel model) { mPolicyModel = model; }

void cNeuralNet::InitIdentityScales()
{
	if (mInputSize > 0)
	{
		if (mInputOffset.size() != mInputSize) mInputOffset = Eigen::VectorXd::Zero(mInputSize);
		if (mInputScale.size() != mInputSize) mInputScale = Eigen::VectorXd::Ones(mInputSize);
	}
	if (mOutputSize > 0)
	{
		if (mOutputOffset.size() != mOutputSize) mOutputOffset = Eigen::VectorXd::Zero(mOutputSize);
		if (mOutputScale.size() != mOutputSize) mOutputScale = Eigen::VectorXd::Ones(mOutputSize);
	}
}

Eigen::VectorXd cNeuralNet::RunClassic(const Eigen::VectorXd& x) const
{
	auto it = mQTable.find(BuildQKey(x));
	if (it != mQTable.end())
	{
		return it->second;
	}
	return Eigen::VectorXd::Zero(mOutputSize);
}

Eigen::VectorXd cNeuralNet::RunTorch(const Eigen::VectorXd& x) const
{
#if AP_DTRL_HAS_TORCH
	if (mHasModule)
	{
		torch::NoGradGuard guard;
		torch::Tensor input = torch::from_blob((double*)x.data(), {1, x.size()}, torch::kFloat64).clone().to(torch::kFloat32);
		auto output = mModule.forward({input}).toTensor().squeeze(0).to(torch::kCPU).to(torch::kFloat64);
		Eigen::VectorXd y(output.numel());
		std::memcpy(y.data(), output.data_ptr<double>(), sizeof(double) * static_cast<size_t>(output.numel()));
		return y;
	}
#endif
	return Eigen::VectorXd::Zero(mOutputSize);
}

Eigen::VectorXd cNeuralNet::ApplyAttention(const Eigen::VectorXd& x) const
{
	if (x.size() == 0)
	{
		return x;
	}

	Eigen::VectorXd weights = x.array().exp();
	double sum = weights.sum();
	if (sum > 1e-8)
	{
		weights /= sum;
	}
	return x.cwiseProduct(weights);
}

long long cNeuralNet::BuildQKey(const Eigen::VectorXd& x) const
{
	long long h = 1469598103934665603LL;
	for (int i = 0; i < x.size(); ++i)
	{
		const long long q = static_cast<long long>(std::llround(x[i] * 100.0));
		h ^= q + 0x9e3779b97f4a7c15LL + (h << 6) + (h >> 2);
	}
	return h;
}

cNeuralNet::ePolicyModel cNeuralNet::ParsePolicyModel(const std::string& model_name)
{
	if (model_name == "classic_rl")
	{
		return ePolicyModel::eClassicRL;
	}
	if (model_name == "deep_rl_attention")
	{
		return ePolicyModel::eDeepRLAttention;
	}
	return ePolicyModel::eDeepRL;
}
