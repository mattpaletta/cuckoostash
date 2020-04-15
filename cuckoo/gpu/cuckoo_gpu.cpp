#include "cuckoo/cuckoo.hpp"
#include "cuckoo/array.hpp"
#include "cuckoo/pcg_random.hpp"

// Add in the kernels.
#include "cuckoo_gpu.cu"

template<>
Cuckoo<cuckoo::CudaBackend>::Cuckoo() {
	for (Entry i = 0; i < CUCKOO_SIZE; ++i) {
		this->ccuckoo[i] = 0;
	}
	for (Entry i = 0; i < STASH_SIZE; ++i) {
		this->cstash[i] = 0;
	}

	auto r1 = cudaMalloc((void **) &gcuckoo, CUCKOO_SIZE * sizeof(Entry));
	auto r2 = cudaMalloc((void **) &gstash, STASH_SIZE * sizeof(Entry));
	cudaMemcpy(gcuckoo, this->ccuckoo, CUCKOO_SIZE * sizeof(Entry), cudaMemcpyHostToDevice);
	cudaMemcpy(gstash, this->cstash, CUCKOO_SIZE * sizeof(Entry), cudaMemcpyHostToDevice);
}

template<>
Cuckoo<cuckoo::CudaBackend>::~Cuckoo() {
	cudaFree(gcuckoo);
	cudaFree(gstash);
}

template<>
int Cuckoo<cuckoo::CudaBackend>::set(const std::size_t N, int* keys, int* values, int* results) {
	GPUArray<int> g_keys(keys, N);
       	GPUArray<int> g_values(values, N);
	GPUArray<int> g_results(results, N);

	g_keys.to_gpu();
	g_values.to_gpu();
	g_results.to_gpu();
	gpu_set<<<this->grid_size(N), this->block_size()>>>(g_keys.to_gpu(), g_values.to_gpu(), g_results.to_gpu(), N);

	int count_failed = 0;
	for (std::size_t i = 0; i < N; ++i) {
		if (!(bool) (g_results.get_gpu() + i)) {
			std::cout << "Failed to insert item: " << i << std::endl;
			count_failed++;
		}
	}

	return count_failed;
}

void Cuckoo<cuckoo::CudaBackend>::get(const std::size_t N, int* keys, int* results) {
	GPUArray<int> g_keys(keys, N);
	GPUArray<int> g_results(results, N);

	gpu_get<<<this->grid_size(N), this->block_size()>>>(g_keys.to_gpu(), g_results.to_gpu(), N);

	const auto from_gpu = g_results.get_gpu();
	for (std::size_t i = 0; i < N; ++i) {
		results[i] = from_gpu[i];
	}
	return from_gpu;
}
