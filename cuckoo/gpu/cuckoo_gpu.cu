#include "cuckoo/cuckoo.hpp"
#include "cuckoo/array.hpp"
#include "cuckoo/pcg_random.hpp"

// Add in the kernels.
#include "cuckoo_gpu_impl.cu"

Cuckoo<cuckoo::CudaBackend>::Cuckoo(const std::size_t N, const std::size_t stash_size, const std::size_t num_hash_functions) {
    for (cuckoo::Entry i = 0; i < cuckoo::CUCKOO_SIZE; ++i) {
		this->cuckoo.get_cpu()[i] = 0;
	}
	for (cuckoo::Entry i = 0; i < cuckoo::STASH_SIZE; ++i) {
		this->stash.get_cpu()[i] = 0;
	}

	this->cuckoo.to_gpu();
	this->stash.to_gpu();
}

Cuckoo<cuckoo::CudaBackend>::~Cuckoo() {
}

int Cuckoo<cuckoo::CudaBackend>::set(const std::size_t& N, int* keys, int* values, int* results) {
	GPUArray<int> g_keys(keys, N);
	GPUArray<int> g_values(values, N);
	GPUArray<int> g_results(results, N);

	for (int i = 0; i < N; ++i) {
		g_results.get_cpu()[i] = 1;
	}

	gpu_set<<<this->grid_size(N), this->block_size()>>>(g_keys.to_gpu(), g_values.to_gpu(), g_results.to_gpu(), N, this->cuckoo.to_gpu(), this->stash.to_gpu());
	cudaCheckError()



	int count_failed = 0;
	for (std::size_t i = 0; i < N; ++i) {
		const int value = g_results.get_gpu()[i];
		if ((bool) value) {
			std::cout << "Failed to insert item: " << i << " " << value << std::endl;
			count_failed++;
		}
	}

	const auto from_gpu = this->cuckoo.get_gpu();
	for (std::size_t i = 0; i < cuckoo::CUCKOO_SIZE; ++i) {
		std::cout << from_gpu[i] << " ";
	}

	return count_failed;
}

void Cuckoo<cuckoo::CudaBackend>::get(const std::size_t& N, int* keys, int* results) {
	GPUArray<int> g_keys(keys, N);
	GPUArray<int> g_results(results, N);

	for (int i = 0; i < N; ++i) {
		g_results.get_cpu()[i] = 1;
	}

	gpu_get<<<this->grid_size(N), this->block_size()>>>(g_keys.to_gpu(), g_results.to_gpu(), N, this->cuckoo.to_gpu(), this->stash.to_gpu());
	cudaCheckError()
	const auto from_gpu = g_results.get_gpu();
	for (std::size_t i = 0; i < N; ++i) {
		results[i] = from_gpu[i];
	}
}
