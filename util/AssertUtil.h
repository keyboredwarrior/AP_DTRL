#pragma once

#include <cstdio>
#include <cstdlib>

#define AP_DTRL_ASSERT(cond)                                                   \
	do                                                                          \
	{                                                                           \
		if (!(cond))                                                              \
		{                                                                         \
			std::fprintf(stderr, "Assertion failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
			std::abort();                                                           \
		}                                                                         \
	} while (false)
