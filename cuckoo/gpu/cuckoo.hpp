#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <atomic>

#include "pcg_random.hpp"
//#include <pybind11/pybind11.h>
//namespace py = pybind11;
typedef unsigned long long Entry;

// TODO:// Pass this in at compile_time
constexpr std::size_t MAX_ITERATIONS = 100;

constexpr Entry CUCKOO_SIZE = 1000;
constexpr Entry STASH_SIZE = 100;

//template<std::size_t N = 1024>
class Cuckoo {
public:
	Cuckoo(const std::size_t N = 1024, const std::size_t stash_size = 10, const std::size_t num_hash_functions = 4);
	~Cuckoo();
	int set(const std::size_t& N, int* keys, int* values, int* results);
	void get(const std::size_t& N, int* keys, int* results);
	using key_type = Entry;
	using value_type = key_type;
	using func_type = std::function<key_type(const key_type&)>;
	static PCG::pcg32_random_t rand;

	enum FuncType {
		LINEAR = 1,
		XOR = 2
	};

private:
	std::size_t _N;
	std::size_t stash_size;
	std::vector<func_type> hash_functions;
	func_type stash_hash_function;
	// TODO: Calculate actual size.
	Entry ccuckoo[CUCKOO_SIZE];
	Entry cstash[STASH_SIZE];

	std::vector<func_type> get_all_hash_functions(const std::size_t& full_table_size, const std::size_t& num_hash_functions);

	func_type get_hash_function(const PCG::pcg32_random_t::result_type& a, const PCG::pcg32_random_t::result_type& b, const std::size_t& p, const std::size_t& stash_size, const FuncType& function);

	func_type get_stash_function(const std::size_t& stash_size);

	func_type get_stash_function(const PCG::pcg32_random_t::result_type& a, const PCG::pcg32_random_t::result_type& b, const std::size_t& p, const std::size_t& stash_size, const std::string& function);

	int block_size() {
		return 256;
	}
	int grid_size(const std::size_t N) {
		return ((N + this->block_size()) / this->block_size());
	}

	key_type get_key(const Cuckoo::key_type& entry) {
		return ((Cuckoo::key_type)((entry) >> 32));
	}

	value_type get_value(Entry entry, key_type key) {
		return entry - this->get_key(key);
	}
};

bool add();
bool add_array();
bool add_new();

/*
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
*/
