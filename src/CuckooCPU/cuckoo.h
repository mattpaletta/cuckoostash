#ifndef PROJECT_CUCKOO_H
#define PROJECT_CUCKOO_H

#include <vector>
#include <string>
#include <functional>

typedef unsigned long long Entry;


class Cuckoo {
private:
    const Entry SLOT_EMPTY = 0xffffffff;
    const Entry KEY_EMPTY = SLOT_EMPTY;

    int MAX_ITERATIONS;
    long max_size_chaining;

    std::vector<Entry> cuckoo_values;
    std::vector<Entry> stash_values;

    std::vector<std::function<unsigned long(Entry)>> hash_functions;
    std::function<unsigned long(Entry)> stash_hash_functions;

    std::vector<std::function<unsigned long(Entry)>> get_all_hash_tables(int, int);
    std::function<unsigned long(Entry)> get_stash_function(int);

    std::function<unsigned long(Entry)> get_next_hash_function(const long, const long, const long, const long, const std::string);

public:
    Cuckoo(unsigned int, int, int);
    Entry get(Entry key);
    bool set(Entry key, Entry value);
};

#endif