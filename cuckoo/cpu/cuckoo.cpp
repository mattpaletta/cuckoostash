#include <cmath>
#include "cuckoo/gpu/cuckoo.hpp"

bool add() {
	return true;
}

bool add_new() {
	return true;
}

bool add_array() {
	return true;
}

Cuckoo::Cuckoo(const std::size_t N, const std::size_t stash_size, const std::size_t num_hash_functions) {
	assert(N > 0);
	assert(num_hash_functions > 0);
	assert(stash_size < N);
	this->_N = N;
	const std::size_t full_table_size = (N * 1.25) + 1;
	using hash_type = long long;
	const std::size_t max_size_chaining = 7 * (log(N) / log(2));
	this->stash_size = pow(stash_size, 2) + 1; // 101
	this->hash_functions = this->get_all_hash_functions(full_table_size, num_hash_functions);
	this->stash_hash_function = this->get_stash_function(this->stash_size);
	constexpr auto SLOT_EMPTY = 0;
	for (std::size_t i = 0; i < CUCKOO_SIZE; ++i) {
		this->ccuckoo[i] = SLOT_EMPTY;
	}

	for (std::size_t i = 0; i < STASH_SIZE; ++i) {
		this->cstash[i] = SLOT_EMPTY;
	}
}

Cuckoo::~Cuckoo() {}

Cuckoo::func_type Cuckoo::get_hash_function(const PCG::pcg32_random_t::result_type& a, const PCG::pcg32_random_t::result_type& b, const std::size_t& p, const std::size_t& stash_size, const FuncType& function) {
	switch (function) {
	case FuncType::LINEAR:
		return [a, b, p, stash_size](const Cuckoo::key_type& k) {
			return ((((a * k) + b) % p) % stash_size);
		};
	case FuncType::XOR:
		return [a, b, p, stash_size](const Cuckoo::key_type& k) {
			return ((((a ^ k) + b) % p) % stash_size);
		};

	}
}

std::vector<Cuckoo::func_type> Cuckoo::get_all_hash_functions(const std::size_t& full_table_size, const std::size_t& num_hash_functions) {
	PCG::pcg32_random_t rand;
	auto rand_range = [&rand](const std::size_t& min, const std::size_t& max) {
		return (rand() % max) + min;
	};
	constexpr auto p = 4'294'967'291;
	std::vector<Cuckoo::func_type> out;
	for (std::size_t i = 0; i < num_hash_functions; ++i) {
		const auto a = rand_range(1, p);
		const auto b = rand_range(0, p);
		out.emplace_back(this->get_hash_function(a, b, p, full_table_size, FuncType::XOR));
	}

	return out;
}

Cuckoo::func_type Cuckoo::get_stash_function(const std::size_t& stash_size) {
	PCG::pcg32_random_t rand;
	auto rand_range = [&rand](const std::size_t& min, const std::size_t& max) {
		return (rand() % max) + min;
	};
	constexpr std::size_t p = 334'214'459;
	const auto a = rand_range(1, p);
	const auto b = rand_range(0, p);

	return this->get_hash_function(a, b, p, stash_size, FuncType::LINEAR);
}

void Cuckoo::get(const std::size_t& N, int* keys, int* results) {
	auto get_item = [this, &N](const Cuckoo::key_type& key) {
		constexpr Entry kEntryNotFound = 0;
		auto entry = this->ccuckoo[this->hash_functions[0](key)];
		for (auto location = 0; location < this->hash_functions.size(); location++) {
			entry = this->ccuckoo[this->hash_functions[location](key)];
			if (entry == kEntryNotFound) {
				// Once we hit a blank, we know that's it.
				return (value_type) 0;
			}
			if (this->get_key(entry) == key) {
				break;
			}
		}

		// Still haven't found it, fetch from stash
		if (entry == kEntryNotFound) {
			entry = this->cstash[this->stash_hash_function(key)];
			if (entry == kEntryNotFound) {
				return (value_type) 0;
			}
		}

		return this->get_value(entry, key);
	};
	for (std::size_t i = 0; i < N; ++i) {
		results[i] = get_item(keys[i]);
	}
}

int Cuckoo::set(const std::size_t& N, int* keys, int* values, int* results) {
	auto set_item = [this, &N](const Cuckoo::key_type& key, const Cuckoo::value_type& value) {
		auto swap = [this](const std::size_t& location, const Cuckoo::key_type& entry) {
			const auto curr_item = this->ccuckoo[location];
			this->ccuckoo[location] = entry;
			return curr_item;
		};

		Cuckoo::key_type curr_key = key;

		Cuckoo::key_type entry = (curr_key << 32) + value;
		std::size_t location_var = 0;

		for (std::size_t i = 0; i < MAX_ITERATIONS; ++i) {
			const auto location = this->hash_functions[location_var](curr_key);
			std::swap(this->ccuckoo[location], entry);
			curr_key = this->get_key(entry);

			if (entry == 0) {
				// Empty
				return 0;
			}

			location_var = (location_var + 1) % this->hash_functions.size();
		}

		const auto stash_slot = this->stash_hash_function(key);
		if (this->cstash[stash_slot] == 0) {
			this->cstash[stash_slot] = entry;
			return 0;
		} else {
			// No room in the stash.
			return 1;
		}
	};

	int num_failed = 0;
	for (std::size_t i = 0; i < N; ++i) {
		num_failed  += set_item(keys[i], values[i]);
	}

	std::cout << "Printing array" << std::endl;
	std::cout << "[";
	for (std::size_t i = 0; i < CUCKOO_SIZE; ++i) {
		std::cout << this->ccuckoo[i] << ", ";
	}
	std::cout << "]" << std::endl;

	return num_failed;
}
