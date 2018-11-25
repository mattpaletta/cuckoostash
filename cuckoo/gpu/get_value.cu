typedef unsigned long long Entry;
const Entry SLOT_EMPTY = (0xffffffff, 0);
const Entry KEY_EMPTY = SLOT_EMPTY;
#define get_key(entry) ((unsigned)((entry) >> 32));
//#define CUCKOO_SIZE %(CUCKOO_SIZE)s;
//#define STASH_SIZE %(STASH_SIZE)s;

#define CUCKOO_SIZE 1000;
#define STASH_SIZE 101;

// TODO:// Pass this in at compile_time
const int MAX_ITERATIONS = 100;

__constant__ int *hash_functions[4];

__device__ Entry *cuckoo[1000];
__device__ Entry *stash[101];

#define hash_function_1(entry) 1;
#define hash_function_2(entry) 1;
#define hash_function_3(entry) 1;
#define hash_function_4(entry) 1;

#define stash_hash_function(entry) 1;


__device__ Entry get_value(Entry entry) {
    return ((unsigned)((entry) & SLOT_EMPTY));
}

__device__ Entry get_helper(int key) {
    const Entry kEntryNotFound = SLOT_EMPTY;

    // Compute all possible locations for the key.
    unsigned location_1 = hash_function_1(key);
    unsigned location_2 = hash_function_2(key);
    unsigned location_3 = hash_function_3(key);
    unsigned location_4 = hash_function_4(key);

    Entry entry = *cuckoo[location_1];

    // Keep checking locations
    // are checked, or if an empty slot is found. Entry entry ;
    entry = *cuckoo[location_1];
    int current_key = get_key(entry);
    if (current_key != key) {
        if (current_key == kEntryNotFound) {
            return kEntryNotFound;
        }

        entry = *cuckoo[location_2];
        int current_key = get_key(entry);

        if (current_key != key) {
            if (current_key == kEntryNotFound) {
                return kEntryNotFound;
            }

            entry = *cuckoo[location_3];
            int current_key = get_key(entry);

            if (current_key != key) {
                if (current_key == kEntryNotFound) {
                    return kEntryNotFound;
                }

                entry = *cuckoo[location_4];
                int current_key = get_key(entry);

                if (current_key != key) {
                    return kEntryNotFound;
                }
            }
        }
    }

    return get_value(entry);
}

__device__ bool set_helper(Entry key, Entry value) {
    Entry entry = (((Entry) key) << 32) + value;

    // Repeat the insertion process while the thread still has an item.
    unsigned location = hash_function_1(key);

    for (int its = 0; its < MAX_ITERATIONS; its++) {
        // Insert the new item and check for an eviction
        entry = atomicExch(stash[location], entry);
        key = get_key(entry);
        if (key == KEY_EMPTY) {
            return true;
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
    Entry replaced_entry = atomicCAS((unsigned long long *)*stash[slot], SLOT_EMPTY, entry);
    return (replaced_entry == SLOT_EMPTY);
}

// TODO:// Rewrite without global memory
__device__ void fetch_value_from_table(int *keys, Entry *values, Entry *results) {
    int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
    int key = keys[thread_index];

    results[thread_index] = get_helper(key);
}

__device__ void set_value_in_table(Entry *keys, Entry *values, bool *results) {
    int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;

    // Load up the key-value pair into a 64-bit entry
    unsigned key = keys[thread_index];
    unsigned value = values[thread_index];

    results[thread_index] = set_helper(key, value);
}