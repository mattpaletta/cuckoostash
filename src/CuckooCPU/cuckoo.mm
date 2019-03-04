#include "cuckoo.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>

#ifdef CUCKOO_SUPPORT_METAL
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#endif

#define get_key(entry) ((unsigned) ((entry) >> 32))
#define get_value(entry) (entry & KEY_EMPTY)

Cuckoo::Cuckoo(unsigned int N, int stash_size, int num_hash_functions) {
    auto full_table_size = (N * 1.25) + 1;
    this->max_size_chaining = (long) (7 * (log(N) / log(2)));

    auto max_stash_size = (unsigned long) pow(stash_size, 2) + 1;

    this->cuckoo_values = {};
    this->stash_values = {};


    this->cuckoo_values.reserve(N);
    this->stash_values.reserve(max_stash_size);

    std::fill(this->cuckoo_values.begin(),
            this->cuckoo_values.begin() + N,
            KEY_EMPTY);
    std::fill(this->cuckoo_values.begin(),
            this->cuckoo_values.begin() + max_stash_size,
            KEY_EMPTY);

    this->hash_functions = this->get_all_hash_tables(full_table_size, num_hash_functions);
    this->stash_hash_functions = this->get_stash_function(stash_size);
}

std::vector<std::function<unsigned long(Entry)>> Cuckoo::get_all_hash_tables(int table_size, int num_hash_functions) {
    const long p = 4294967291;


    std::vector<std::function<unsigned long(Entry)>> out_vector;

    for (int i = 0; i < num_hash_functions; i++) {
        long a = random() % p + 1;
        long b = random() % p + 0;

        out_vector.push_back(
                this->get_next_hash_function(a, b, p, table_size, "lienar"));
    }

    return out_vector;
}

std::function<unsigned long(Entry)> Cuckoo::get_stash_function(int stash_size) {
    const long p = 334214459;

    long a = random() % p + 1;
    long b = random() % p + 0;

    return this->get_next_hash_function(a, b, p, stash_size, "linear");
}

std::function<unsigned long(Entry)> Cuckoo::get_next_hash_function(const long a, const long b, const long p, const long s_t, const std::string function) {
    const auto linear_func = [&a, &b, &p, &s_t](Entry k) {
        return (((a * k) + b) % p) % s_t;
    };

    const auto xor_func = [&a, &b, &p, &s_t](Entry k) {
        return (((a ^ k) + b) % p) % s_t;
    };

    if (function == "linear") {
        return linear_func;
    } else if (function == "xor") {
        return xor_func;
    } else {
        return nullptr;
    }
}

Entry Cuckoo::get(Entry key) {
    auto location = this->hash_functions.at(0)(key);
    auto entry = this->cuckoo_values.at(location);
    for (int location_i = 0; location_i < this->hash_functions.size(); location_i++) {
        entry = this->cuckoo_values.at(location_i);
        if ((get_key(entry)) == key) {
            break;
        } else if ((get_key(entry)) == KEY_EMPTY) {
            return Cuckoo::KEY_EMPTY;
        }
    }

    return get_value(entry);
}

bool Cuckoo::set(Entry key, Entry value) {
    Entry entry = (key << 32) + value;
    long location = (long) this->hash_functions.at(0)(key);

    for (int i = 0; i < this->max_size_chaining; i++) {
        auto temp = entry;
        entry = this->cuckoo_values.at(location);
        this->cuckoo_values.at(location) = temp;
        auto curr_key = get_key(entry);

        if (curr_key == KEY_EMPTY) {
            return true;
        }

        for (int j = 0; j < this->hash_functions.size(); j++) {
            if (location == this->hash_functions.at(j)(key)) {
                location = this->hash_functions.at((j + 1) % this->hash_functions.size())(key);
                break;
            }
        }
    }

    auto slot = this->stash_hash_functions(key);
    if (this->stash_values.at(slot) == KEY_EMPTY) {
        this->stash_values.at(slot) = entry;
    }

    return (slot == KEY_EMPTY);
}

void TestMetal() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    auto shadersSrc = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void sqr(
            const device float *vIn [[ buffer(0) ]],
            device float *vOut [[ buffer(1) ]],
            uint id[[ thread_position_in_grid ]]) {
                vOut[id] = vIn[id] * vIn[id];
            }
        );
        """;

    auto library = [device newLibraryWithSource:@shadersSrc options: MTLCompileOptions() error:nullptr];
    assert(library);
    auto sqrFunc = [library newFunctionWithName:@"sqr"];
    assert(sqrFunc);

    auto computePipelineState = [device newComputePipelineStateWithFunction:sqrFunc error:nullptr];
    assert(computePipelineState);

    auto commandQueue = device.newCommandQueue();
    assert(commandQueue);

    const uint32_t dataCount = 6;

    auto inBuffer = [device newBufferWithLength:sizeof(float) * dataCount options:MTLResourceOptions::MTLResourceStorageModeManaged]
    assert(inBuffer);

    auto outBuffer = [device newBufferWithLength:sizeof(float) * dataCount options:MTLResourceOptions::MTLResourceStorageModeManaged]
    assert(outBuffer);

    for (uint32_t i=0; i<4; i++) {
        // update input data
        float* inData = static_cast<float*>(inBuffer.GetContents());
        for (uint32_t j=0; j<dataCount; j++) {
            inData[j] = 10 * i + j;
        }
        inBuffer.DidModify(ns::Range(0, sizeof(float) * dataCount));
    }
//    mtlpp::CommandBuffer commandBuffer = commandQueue.CommandBuffer();
//    assert(commandBuffer);
//
//    mtlpp::ComputeCommandEncoder commandEncoder = commandBuffer.ComputeCommandEncoder();
//    commandEncoder.SetBuffer(inBuffer, 0, 0);
//    commandEncoder.SetBuffer(outBuffer, 0, 1);
//    commandEncoder.SetComputePipelineState(computePipelineState);
//    commandEncoder.DispatchThreadgroups(
//            mtlpp::Size(1, 1, 1),
//            mtlpp::Size(dataCount, 1, 1));
//    commandEncoder.EndEncoding();
//
//    mtlpp::BlitCommandEncoder blitCommandEncoder = commandBuffer.BlitCommandEncoder();
//    blitCommandEncoder.Synchronize(outBuffer);
//    blitCommandEncoder.EndEncoding();
//
//    commandBuffer.Commit();
//    commandBuffer.WaitUntilCompleted();

    // read the data {
    float* inData = static_cast<float*>(inBuffer.GetContents());
    float* outData = static_cast<float*>(outBuffer.GetContents());
    for (uint32_t j=0; j<dataCount; j++)
        printf("sqr(%g) = %g\n", inData[j], outData[j]);
}

//__device__ Entry get_value(Entry entry) {
//    return ((unsigned)((entry) & SLOT_EMPTY));
//}

//__global__ void fetch(Entry *cuckoo, Entry *stash, int *keys, int *results) {
//    int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
//    int key = keys[thread_index];
//
//    const Entry kEntryNotFound = SLOT_EMPTY;
//
//    // Compute all possible locations for the key.
//    unsigned location_1 = hash_function_1(key);
//    unsigned location_2 = hash_function_2(key);
//    unsigned location_3 = hash_function_3(key);
//    unsigned location_4 = hash_function_4(key);
//
//    Entry entry = cuckoo[location_1];
//
//    // Keep checking locations
//    // are checked, or if an empty slot is found. Entry entry ;
//    entry = cuckoo[location_1];
//    int current_key = get_key(entry);
//    if (current_key != key) {
//        if (current_key == kEntryNotFound) {
//            results[thread_index] = kEntryNotFound;
//            return;
//        }
//
//        entry = cuckoo[location_2];
//        int current_key = get_key(entry);
//
//        if (current_key != key) {
//            if (current_key == kEntryNotFound) {
//                results[thread_index] = kEntryNotFound;
//                return;
//            }
//
//            entry = cuckoo[location_3];
//            int current_key = get_key(entry);
//
//            if (current_key != key) {
//                if (current_key == kEntryNotFound) {
//                    results[thread_index] = kEntryNotFound;
//                    return;
//                }
//
//                entry = cuckoo[location_4];
//                int current_key = get_key(entry);
//
//                if (current_key != key) {
//                    results[thread_index] = kEntryNotFound;
//                    return;
//                }
//            }
//        }
//    }
//
//    results[thread_index] = get_value(entry);
//}

//__global__ void set(Entry *cuckoo, Entry *stash, Entry *keys, Entry *values, int *results) {
//    int thread_index = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y;
//
//    // Load up the key-value pair into a 64-bit entry
//    unsigned key = keys[thread_index];
//    unsigned value = values[thread_index];
//    Entry entry = (((Entry) key) << 32) + value;
//
//    // Repeat the insertion process while the thread still has an item.
//    unsigned location = hash_function_1(key);
//
//    for (int its = 0; its < MAX_ITERATIONS; its++) {
//        // Insert the new item and check for an eviction
//        entry = atomicExch(&stash[location], entry);
//        key = get_key(entry);
//        if (key == KEY_EMPTY) {
//            results[thread_index] = 0;
//            return;
//        }
//
//        // If an item was evicted, figure out where to reinsert the entry.
//        unsigned location_1 = hash_function_1(key);
//        unsigned location_2 = hash_function_2(key);
//        unsigned location_3 = hash_function_3(key);
//        unsigned location_4 = hash_function_4(key);
//
//        if (location == location_1) {
//            location = location_2;
//        } else if (location == location_2) {
//            location = location_3;
//        } else if (location == location_3) {
//            location = location_4;
//        } else {
//            location = location_1;
//        }
//    }
//
//    // Try the stash. It succeeds if the stash slot it needs is empty.
//    unsigned slot = stash_hash_function(key);
//    Entry replaced_entry = atomicCAS((Entry*) &stash[slot], SLOT_EMPTY, entry);
//    results[thread_index] = (int) replaced_entry == SLOT_EMPTY;
//}