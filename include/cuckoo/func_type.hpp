//
// Created by mattp on 4/15/2020.
//
#pragma once
#include <functional>

namespace cuckoo {



    typedef unsigned long long Entry;

    enum FuncType {
        LINEAR = 1,
        XOR = 2
    };

    using key_type = Entry;
    using value_type = key_type;

    using func_type = std::function<key_type(const key_type&)>;


    // TODO:// Pass this in at compile_time
    constexpr std::size_t MAX_ITERATIONS = 100;
    constexpr cuckoo::Entry CUCKOO_SIZE = 1000;
    constexpr cuckoo::Entry STASH_SIZE = 100;
}