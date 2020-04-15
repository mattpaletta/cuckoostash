#pragma once

namespace cuckoo {

struct Backend {
	static const int device_id = 0;
};

struct CudaBackend final : public Backend {
	static const int device_id = 1;
};

struct CpuBackend final : public Backend {
	static const int device_id = 2;
};

using AnyBackend = Backend;

};
