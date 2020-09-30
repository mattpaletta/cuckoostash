#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <atomic>
#include <functional>

#include "func_type.hpp"
#include "device.hpp"
#include "pcg_random.hpp"

template<typename backend = cuckoo::AnyBackend>
class Cuckoo {
public:
	Cuckoo(const std::size_t N = 1024, const std::size_t stash_size = 10, const std::size_t num_hash_functions = 4);
	~Cuckoo();
	virtual int set(const std::size_t& N, int* keys, int* values, int* results);
	virtual void get(const std::size_t& N, int* keys, int* results);

	static PCG::pcg32_random_t rand;
};
