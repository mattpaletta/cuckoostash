#pragma once
#include "cuckoo_base.hpp"

template<>
class Cuckoo<cuckoo::CpuBackend> {
public:
    Cuckoo(const std::size_t N = 1024, const std::size_t stash_size = 10, const std::size_t num_hash_functions = 4);
    ~Cuckoo() = default;

    void get(const std::size_t& N, int* keys, int* results);
    int set(const std::size_t& N, int* keys, int* values, int* results);

    std::vector<cuckoo::func_type> get_all_hash_functions(const std::size_t& full_table_size, const std::size_t& num_hash_functions);
    cuckoo::func_type get_hash_function(const PCG::pcg32_random_t::result_type& a, const PCG::pcg32_random_t::result_type& b, const std::size_t& p, const std::size_t& stash_size, const cuckoo::FuncType& function);
    cuckoo::func_type get_stash_function(const std::size_t& stash_size);
    cuckoo::func_type get_stash_function(const PCG::pcg32_random_t::result_type& a, const PCG::pcg32_random_t::result_type& b, const std::size_t& p, const std::size_t& stash_size, const std::string& function);

private:
    std::size_t _N;
    std::size_t stash_size;
    std::vector<cuckoo::func_type> hash_functions;
    cuckoo::func_type stash_hash_function;
    // TODO: Calculate actual size.
    cuckoo::key_type ccuckoo[cuckoo::CUCKOO_SIZE];
    cuckoo::key_type cstash[cuckoo::STASH_SIZE];

    cuckoo::key_type get_key(const cuckoo::key_type& entry) {
        return ((cuckoo::key_type)((entry) >> 32));
    }

    cuckoo::value_type get_value(const cuckoo::key_type& entry, const cuckoo::key_type& key) {
        return entry - this->get_key(key);
    }
};