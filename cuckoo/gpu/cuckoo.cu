#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "array.hpp"
#include "cuckoo.hpp"
#include "pcg_random.hpp"


// Kernel function to add the elements of two arrays
__global__ void add(const int n, const float *x, float *y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    	for (int i = index; i < n; i += stride) {
    		if (i < n) {
        		y[i] = x[i] + y[i];
    		}
	}
}

bool is_same(std::size_t N, float* a, float* b) {
	bool is_same = true;
	for (std::size_t i = 0; i < N; ++i) {
		if (a[i] != b[i]) {
			is_same = false;
		}
	}
	return is_same;
}

__host__ bool add() {
	constexpr int N = 100;
	float *x = new float[N];
	float *y = new float[N];
	
	std::vector<float> z;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));
	
	// initialize x and y arrays on the host
	for (int i = 0; i < N; ++i) {
		const float x_val = 5.0f;
		const float y_val = 2.0f + (float) i;
		x[i] = x_val;
		y[i] = y_val;
		z.push_back(x_val + y_val);
	}

	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);


	// Run kernel on 1M elements on the GPU
	add<<<1, N/ 256>>>(N, x, y);

    	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	
	const bool result = is_same(N, &y[0], &z[0]);
	for (int i = 0; i < N; ++i) {
		std::cout << y[i] << ":" << z[i] << std::endl;
	}

	// Free memory
	cudaFree(x);
	cudaFree(y);
	delete[] x;
	delete[] y;
	return result;
}

__device__ Entry SLOT_EMPTY = (0xffffffff, 0);
#define get_key(entry) ((unsigned)((entry) >> 32));

__device__ Entry gcuckoo[CUCKOO_SIZE];
__device__ Entry gstash[STASH_SIZE];

__device__ Entry hash_function_1(Entry entry) {
	return 1;
}
__device__ Entry hash_function_2(Entry entry) {
	return 1;
}
__device__ Entry hash_function_3(Entry entry) {
	return 1;
}
__device__ Entry hash_function_4(Entry entry) {
	return 1;
}

#define stash_hash_function(entry) 1;

__device__ Entry get_value(Entry entry) {
    return ((unsigned)((entry) & SLOT_EMPTY));
}

__global__ void gpu_get(const int* keys, int* results, const std::size_t N) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
    if (thread_index > N) {
    	results[thread_index] = -2;
	return;
    } else {
    	results[thread_index] = -1;
	return;
    }
    const int key = keys[thread_index];

    const Entry kEntryNotFound = SLOT_EMPTY;

    // Compute all possible locations for the key.
    const unsigned location_1 = hash_function_1(key);
    const unsigned location_2 = hash_function_2(key);
    const unsigned location_3 = hash_function_3(key);
    const unsigned location_4 = hash_function_4(key);

    Entry entry = gcuckoo[location_1];

    // Keep checking locations
    // are checked, or if an empty slot is found. Entry entry ;
    entry = gcuckoo[location_1];
    int current_key = get_key(entry);
    if (current_key != key) {
        if (current_key == kEntryNotFound) {
            results[thread_index] = kEntryNotFound;
            return;
        }

        entry = gcuckoo[location_2];
        int current_key = get_key(entry);

        if (current_key != key) {
            if (current_key == kEntryNotFound) {
                results[thread_index] = kEntryNotFound;
                return;
            }

            entry = gcuckoo[location_3];
            int current_key = get_key(entry);

            if (current_key != key) {
                if (current_key == kEntryNotFound) {
                    results[thread_index] = kEntryNotFound;
                    return;
                }

                entry = gcuckoo[location_4];
                int current_key = get_key(entry);

                if (current_key != key) {
                    results[thread_index] = kEntryNotFound;
                    return;
                }
            }
        }
    }

    results[thread_index] = get_value(entry);
}

__global__ void gpu_set(int* keys, int* values, int* results, const std::size_t N) {
    int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
    if (thread_index > N) {
    	return;
    }
    // Load up the key-value pair into a 64-bit entry
    unsigned key = keys[thread_index];
    unsigned value = values[thread_index];
    Entry entry = (((Entry) key) << 32) + value;

    // Repeat the insertion process while the thread still has an item.
    unsigned location = hash_function_1(key);

    for (int its = 0; its < MAX_ITERATIONS; its++) {
        // Insert the new item and check for an eviction
        entry = atomicExch(&gstash[location], entry);
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
    unsigned slot = stash_hash_function(key);
    Entry replaced_entry = atomicCAS((Entry*) &gstash[slot], SLOT_EMPTY, entry);
    results[thread_index] = (int) replaced_entry == SLOT_EMPTY;
}

Cuckoo::Cuckoo() {
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

Cuckoo::~Cuckoo() {
	cudaFree(gcuckoo);
	cudaFree(gstash);
}

int Cuckoo::set(const std::size_t N, int* keys, int* values, int* results) {
	GPUArray<int> g_keys(keys, N);
       	GPUArray<int> g_values(values, N);
	GPUArray<int> g_results(results, N);

	std::cout << "Sending to GPU" << std::endl;
	g_keys.to_gpu();
	g_values.to_gpu();
	g_results.to_gpu();
	std::cout << "Running command on GPU" << std::endl;
	gpu_set<<<N / 256, 256>>>(g_keys.to_gpu(), g_values.to_gpu(), g_results.to_gpu(), N);
	
	int count_failed = 0;
	for (std::size_t i = 0; i < N; ++i) {
		if (!(bool) (g_results.get_gpu() + i)) {
			std::cout << "Failed to insert item: " << i << std::endl;
			count_failed++;
		}
	}
	
	return count_failed;
}

int* Cuckoo::get(std::size_t N, int* keys, int* results) {
	GPUArray<int> g_keys(keys, N);
	GPUArray<int> g_results(results, N);

	gpu_get<<<N / 256, 256>>>(g_keys.to_gpu(), g_results.to_gpu(), N);

	const auto from_gpu = g_results.get_gpu();
	for (std::size_t i = 0; i < N; ++i) {
		results[i] = from_gpu[i];
	}
	return g_results.get_gpu();
}
