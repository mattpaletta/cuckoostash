#pragma once
#include "cuckoo_base.hpp"
#include "array.hpp"

template<>
class Cuckoo<cuckoo::CudaBackend> {
public:
	Cuckoo(const std::size_t N = 1024, const std::size_t stash_size = 10, const std::size_t num_hash_functions = 4);
	~Cuckoo();

	int set(const std::size_t& N, int* keys, int* values, int* results);
	void get(const std::size_t& N, int* keys, int* results);

private:
	GPUArray<cuckoo::key_type> cuckoo{cuckoo::CUCKOO_SIZE};
	GPUArray<cuckoo::key_type> stash{cuckoo::STASH_SIZE};

	static int block_size() {
		return 256;
	}

	static int grid_size(const std::size_t N) {
		return ((N + Cuckoo<cuckoo::CudaBackend>::block_size()) / Cuckoo<cuckoo::CudaBackend>::block_size());
	}
};
