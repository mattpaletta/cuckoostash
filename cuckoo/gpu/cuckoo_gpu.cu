#include <cuda_runtime.h>

__host__ bool add_array() {
	constexpr auto N = 1<<20;
	GPUArray<float> a(N);
	GPUArray<float> b(N);
	GPUArray<float> out(N);

	float *z = new float[N];

	// Initialize host arrays
	for (int i = 0; i < N; i++){
		a.get_cpu()[i] = 1.0f;
		b.get_cpu()[i] = 2.0f;
		z[i] = a.get_cpu()[i] + b.get_cpu()[i];
	}

	// Executing kernel
	int block_size = 256;
	int grid_size = ((N + block_size) / block_size);
	a.to_gpu();
	b.to_gpu();
	out.to_gpu();
	vector_add<<<grid_size, block_size>>>(out.to_gpu(), a.to_gpu(), b.to_gpu(), N);

	// Verification
	auto result = is_same(N, out.get_gpu(), z);

	delete[] z;
	return result;
}

__device__ Entry SLOT_EMPTY = (0xffffffff, 0);
#define get_key(entry) ((unsigned)((entry) >> 32));

__device__ Entry gcuckoo[CUCKOO_SIZE];
__device__ Entry gstash[STASH_SIZE];

__device__ Entry hash_function_1(Entry entry) {
	return 0;
}
__device__ Entry hash_function_2(Entry entry) {
	return 1;
}
__device__ Entry hash_function_3(Entry entry) {
	return 2;
}
__device__ Entry hash_function_4(Entry entry) {
	return 2;
}

#define stash_hash_function(entry) 1;

__device__ Entry get_value(Entry entry) {
    return ((unsigned)((entry) & SLOT_EMPTY));
}

__global__ void gpu_get(const int* keys, int* results, const std::size_t N) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
    if (thread_index > N) {
	return;
    } else {
	results[thread_index] = SLOT_EMPTY;
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
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
    if (thread_index > N) {
    	return;
    }
    // Load up the key-value pair into a 64-bit entry
    unsigned key = keys[thread_index];
    const unsigned value = values[thread_index];
    Entry entry = (((Entry) key) << 32) + value;

    // Repeat the insertion process while the thread still has an item.
    unsigned location = hash_function_1(key);

    for (int its = 0; its < MAX_ITERATIONS; its++) {
        // Insert the new item and check for an eviction
        entry = atomicExch(&gcuckoo[location], entry);
        key = get_key(entry);
        if (key == SLOT_EMPTY) {
            results[thread_index] = key;
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
