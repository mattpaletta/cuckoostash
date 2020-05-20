#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <atomic>

#include "device.hpp"
#include "pcg_random.hpp"

template<typename backend = cuckoo::AnyBackend>
class Cuckoo {
public:
	Cuckoo(const std::size_t& N = 1024, const std::size_t& stash_size = 10, const std::size_t& max_iterations = 100, const std::size_t& num_hash_functions = 4);
	~Cuckoo();
	virtual int set(const std::size_t& N, int* keys, int* values, int* results);
	virtual void get(const std::size_t& N, int* keys, int* results);
	using key_type = unsigned long long;
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
	// key_type ccuckoo[];
	// key_type cstash[];

	std::vector<func_type> get_all_hash_functions(const std::size_t& full_table_size, const std::size_t& num_hash_functions);

	func_type get_hash_function(const PCG::pcg32_random_t::result_type& a, const PCG::pcg32_random_t::result_type& b, const std::size_t& p, const std::size_t& stash_size, const FuncType& function);

	func_type get_stash_function(const std::size_t& stash_size);

	func_type get_stash_function(const PCG::pcg32_random_t::result_type& a, const PCG::pcg32_random_t::result_type& b, const std::size_t& p, const std::size_t& stash_size, const std::string& function);

	key_type get_key(const key_type& entry) {
		return ((Cuckoo::key_type)((entry) >> 32));
	}

	value_type get_value(const key_type& entry, const key_type& key) {
		return entry - this->get_key(key);
	}
};
