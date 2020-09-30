#include <cuckoo/cuckoo.hpp>

Cuckoo<cuckoo::Backend>::Cuckoo(const std::size_t N, const std::size_t stash_size, const std::size_t num_hash_functions) {}

Cuckoo<cuckoo::Backend>::~Cuckoo() {}

int Cuckoo<cuckoo::Backend>::set(const std::size_t& N, int* keys, int* values, int* results) {
	return cuckoo_impl::set(N, keys, values, results);
}

void Cuckoo<cuckoo::Backend>::get(const std::size_t& N, int* keys, int* results) {
	cuckoo_impl::get(N, keys, results);
}
