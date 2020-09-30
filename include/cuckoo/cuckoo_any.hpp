#pragma once
#include "cuckoo_base.hpp"


template<>
#if __has_include(<cuda_runtime.h>)
class Cuckoo<cuckoo::AnyBackend> : public Cuckoo<cuckoo::CudaBackend> {
#else
template<>
class Cuckoo<cuckoo::AnyBackend> : public Cuckoo<cuckoo::CpuBackend> {
#endif
public:
	Cuckoo(const std::size_t N = 1024, const std::size_t stash_size = 10, const std::size_t num_hash_functions = 4);
	~Cuckoo();

	int set(const std::size_t& N, int* keys, int* values, int* results);
	void get(const std::size_t& N, int* keys, int* results);
private:
#if __has_include(<cuda_runtime.h>)
    using cuckoo_impl = Cuckoo<cuckoo::CudaBackend>;
#else
    using cuckoo_impl = Cuckoo<cuckoo::CpuBackend>;
#endif
};
