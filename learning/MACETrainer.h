#pragma once

#include "util/MathUtil.h"
#include <limits>

class cMACETrainer
{
public:
	static int CalcNumFrags(int output_size, int frag_size)
	{
		if (frag_size <= 0)
		{
			return 0;
		}
		return output_size / (frag_size + 1);
	}

	static int GetMaxFragIdx(const Eigen::VectorXd& params, int num_action_frags)
	{
		int best_idx = 0;
		double best_val = -std::numeric_limits<double>::infinity();
		for (int i = 0; i < num_action_frags; ++i)
		{
			double v = GetVal(params, i);
			if (v > best_val)
			{
				best_val = v;
				best_idx = i;
			}
		}
		return best_idx;
	}

	static double GetMaxFragVal(const Eigen::VectorXd& params, int num_action_frags)
	{
		if (num_action_frags <= 0)
		{
			return 0;
		}
		return GetVal(params, GetMaxFragIdx(params, num_action_frags));
	}

	static void GetFrag(const Eigen::VectorXd& params, int num_action_frags, int frag_size, int frag_idx, Eigen::VectorXd& out_frag)
	{
		const int frag_offset = num_action_frags + frag_idx * frag_size;
		out_frag = params.segment(frag_offset, frag_size);
	}

	static void SetFrag(const Eigen::VectorXd& frag, int frag_idx, int num_action_frags, int frag_size, Eigen::VectorXd& out_params)
	{
		const int frag_offset = num_action_frags + frag_idx * frag_size;
		out_params.segment(frag_offset, frag_size) = frag;
	}

	static double GetVal(const Eigen::VectorXd& params, int frag_idx)
	{
		return params[frag_idx];
	}

	static void SetVal(double val, int frag_idx, Eigen::VectorXd& out_params)
	{
		out_params[frag_idx] = val;
	}

	static void SetActionFragIdx(int action_idx, Eigen::VectorXd& out_action)
	{
		if (out_action.size() > 0)
		{
			out_action[0] = static_cast<double>(action_idx);
		}
	}

	static void SetActionFrag(const Eigen::VectorXd& frag, Eigen::VectorXd& out_action)
	{
		if (out_action.size() >= 1 + frag.size())
		{
			out_action.segment(1, frag.size()) = frag;
		}
	}
};
