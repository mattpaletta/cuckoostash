#pragma once
#include "cuckoo_base.hpp"
#include "cuckoo_any.hpp"
#include "cuckoo_cpu.hpp"

#if __has_include(<cuda_runtime.h>)
	#include "cuckoo_gpu.hpp"
#endif
