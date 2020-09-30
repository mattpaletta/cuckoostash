#pragma once
#include "cuckoo_base.hpp"
#include "cuckoo_cpu.hpp"


#if __has_include(<cuda_runtime.h>)
	#include "cuckoo_gpu.hpp"
#endif

// This must come last because it can inherit from the CPU or GPU.
#include "cuckoo_any.hpp"