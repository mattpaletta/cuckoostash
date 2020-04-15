#pragma once

namespace cuckoo {

struct Backend {
	static const int device_id = 0;
};

#if __has_include(<cuda_runtime.h>)
struct CudaBackend final : public Backend {
	static const int device_id = 1;
};
#endif

struct CpuBackend final : public Backend {
	static const int device_id = 2;
};

using AnyBackend = Backend;

};
