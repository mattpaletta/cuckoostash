#include <cuda_runtime.h>
#include "cuckoo/func_type.hpp"
//
//__host__ bool add_array() {
//	constexpr auto N = 1<<20;
//	GPUArray<float> a(N);
//	GPUArray<float> b(N);
//	GPUArray<float> out(N);
//
//	float *z = new float[N];
//
//	// Initialize host arrays
//	for (int i = 0; i < N; i++){
//		a.get_cpu()[i] = 1.0f;
//		b.get_cpu()[i] = 2.0f;
//		z[i] = a.get_cpu()[i] + b.get_cpu()[i];
//	}
//
//	// Executing kernel
//	int block_size = 256;
//	int grid_size = ((N + block_size) / block_size);
//	a.to_gpu();
//	b.to_gpu();
//	out.to_gpu();
//	vector_add<<<grid_size, block_size>>>(out.to_gpu(), a.to_gpu(), b.to_gpu(), N);
//
//	// Verification
//	auto result = is_same(N, out.get_gpu(), z);
//
//	delete[] z;
//	return result;
//}

// https://gist.github.com/jefflarkin/5390993
//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if (e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(0); \
    }                                                                 \
}

__device__ cuckoo::Entry SLOT_EMPTY = 0;

__device__ cuckoo::Entry hash_function_1(cuckoo::Entry entry) {
	return 0;
}
__device__ cuckoo::Entry hash_function_2(cuckoo::Entry entry) {
	return 1;
}
__device__ cuckoo::Entry hash_function_3(cuckoo::Entry entry) {
	return 2;
}
__device__ cuckoo::Entry hash_function_4(cuckoo::Entry entry) {
	return 2;
}

__device__ cuckoo::Entry stash_hash_function(cuckoo::Entry entry) {
    return 0;
}

__device__ cuckoo::Entry get_key(cuckoo::Entry entry) {
    return ((unsigned)((entry) >> 32));
}

__device__ cuckoo::Entry get_value(cuckoo::Entry entry) {
    return entry - get_key(entry);
}

__global__ void gpu_get(const int* keys, int* results, const std::size_t N, cuckoo::key_type* gcuckoo, cuckoo::key_type* gstash) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
    if (thread_index > N) {
	    return;
    }
    const int key = keys[thread_index];

    cuckoo::Entry entry = gcuckoo[hash_function_1(key)];

    // Compute all possible locations for the key.
    for (int i = 1; i <= 4; ++i) {
        auto location = -1;
        if (i == 1) {
             location = hash_function_1(key);
        } else if (i == 2) {
            location = hash_function_2(key);
        } else if (i == 3) {
            location = hash_function_3(key);
        } else {
            location = hash_function_4(key);
        }

        entry = gcuckoo[location];

        if (entry == SLOT_EMPTY) {
            results[thread_index] = location;
            return;
        }

        if (get_key(entry) == key) {
            break;
        }
    }

    if (entry == SLOT_EMPTY) {
        entry = gstash[stash_hash_function(key)];
        if (entry == SLOT_EMPTY) {
            results[thread_index] = -2;
            return;
        }
    }

    // Keep checking locations
    // are checked, or if an empty slot is found. Entry entry ;
    results[thread_index] = get_value(entry);
}

__global__ void gpu_set(int* keys, int* values, int* results, const std::size_t N, cuckoo::key_type* gcuckoo, cuckoo::key_type* gstash) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
    if (thread_index > N) {
    	return;
    }
    // Load up the key-value pair into a 64-bit entry
    int key = keys[thread_index];
    const auto value = values[thread_index];
    cuckoo::Entry entry = (((cuckoo::Entry) key) << 32) + value;

    // Repeat the insertion process while the thread still has an item.
    auto location = hash_function_1(get_key(entry));

    for (std::size_t its = 0; its < cuckoo::MAX_ITERATIONS; its++) {
        // Insert the new item and check for an eviction
        entry = atomicExch(&gcuckoo[location], entry);
        key = get_key(entry);
        if (key == SLOT_EMPTY) {
            results[thread_index] = 0;
            return;
        }

        // If an item was evicted, figure out where to reinsert the entry.
        unsigned location_1 = hash_function_1(key);
        unsigned location_2 = hash_function_2(key);
        unsigned location_3 = hash_function_3(key);
        unsigned location_4 = hash_function_4(key);

        if (location == location_1) {
            location = location_2;
        } else if (location == location_2) {
            location = location_3;
        } else if (location == location_3) {
            location = location_4;
        } else {
            location = location_1;
        }
    }

    // Try the stash. It succeeds if the stash slot it needs is empty.
    auto slot = stash_hash_function(get_key(entry));
    cuckoo::Entry replaced_entry = atomicCAS((cuckoo::Entry*) &gstash[slot], SLOT_EMPTY, entry);
    results[thread_index] = (int) replaced_entry == SLOT_EMPTY;
}
