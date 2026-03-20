#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "util/MathUtil.h"

#if __has_include(<torch/script.h>)
#include <torch/script.h>
#define AP_DTRL_HAS_TORCH 1
#else
#define AP_DTRL_HAS_TORCH 0
#endif

class cNeuralNet
{
public:
	enum class ePolicyModel
	{
		eClassicRL,
		eDeepRL,
		eDeepRLAttention
	};

	cNeuralNet();

	void Clear();
	bool HasNet() const;
	bool HasLayer(const std::string& layer_name) const;

	void LoadNet(const std::string& net_file);
	void LoadModel(const std::string& model_file);
	void LoadScale(const std::string& scale_file);
	void CopyModel(const cNeuralNet& other);
	void OutputModel(const std::string& out_file) const;

	void Eval(const Eigen::VectorXd& x, Eigen::VectorXd& out_y) const;
	void ForwardInjectNoisePrefilled(const Eigen::VectorXd& noise_mean,
									const Eigen::VectorXd& noise_stdev,
									const std::string& layer_name,
									Eigen::VectorXd& out_y) const;

	void NormalizeInput(Eigen::VectorXd& inout_x) const;
	void NormalizeOutput(Eigen::VectorXd& inout_y) const;

	int GetInputSize() const;
	int GetOutputSize() const;
	const Eigen::VectorXd& GetOutputOffset() const;
	const Eigen::VectorXd& GetOutputScale() const;

	void SetLayerState(const Eigen::VectorXd& state, const std::string& layer_name);
	void GetLayerState(const std::string& layer_name, Eigen::VectorXd& out_state) const;

	ePolicyModel GetPolicyModel() const;
	void SetPolicyModel(ePolicyModel model);

private:
	ePolicyModel mPolicyModel;
	int mInputSize;
	int mOutputSize;
	Eigen::VectorXd mInputOffset;
	Eigen::VectorXd mInputScale;
	Eigen::VectorXd mOutputOffset;
	Eigen::VectorXd mOutputScale;
	std::unordered_map<std::string, Eigen::VectorXd> mLayerState;

#if AP_DTRL_HAS_TORCH
	torch::jit::Module mModule;
	bool mHasModule;
#endif

	std::unordered_map<long long, Eigen::VectorXd> mQTable;

	void InitIdentityScales();
	Eigen::VectorXd RunClassic(const Eigen::VectorXd& x) const;
	Eigen::VectorXd RunTorch(const Eigen::VectorXd& x) const;
	Eigen::VectorXd ApplyAttention(const Eigen::VectorXd& x) const;
	long long BuildQKey(const Eigen::VectorXd& x) const;
	static ePolicyModel ParsePolicyModel(const std::string& model_name);
};
