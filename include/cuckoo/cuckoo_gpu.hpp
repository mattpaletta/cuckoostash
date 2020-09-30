<<<<<<< HEAD
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
=======
#pragma once
#include "cuckoo_base.hpp"

template<>
class Cuckoo<cuckoo::CudaBackend> : public Cuckoo<cuckoo::AnyBackend> {
public:
	Cuckoo(const std::size_t N = 1024, const std::size_t stash_size = 10, const std::size_t num_hash_functions = 4);
	~Cuckoo();

	int set(const std::size_t& N, int* keys, int* values, int* results) override;
	void get(const std::size_t& N, int* keys, int* results) override;

private:
	int static block_size() {
		return 256;
	}

	int static grid_size(const std::size_t& N) {
		const block_size = Cuckoo<cuckoo::CudaBackend>::block_size();
		return ((N + block_size) / block_size);
	}
};
>>>>>>> aad18659ff078ed7ea25632849e2953766b4f5a4
