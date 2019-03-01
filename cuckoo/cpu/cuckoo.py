import random
from math import log
from typing import List, Union

import numpy as np
from numba import prange, jit

KEY_EMPTY = SLOT_EMPTY = [0xffffffff, 0]
ENTRY_NOT_FOUND = -1


class CuckooCpu(object):
    __slots__ = ["_max_size_chaining", "_cuckoo_values", "_stash_values", "_hash_functions",
                 "_stash_hash_function", "_allow_parallel"]

    def __init__(self, N: int, stash_size = 10, num_hash_functions = 4, parallel = True):
        if N <= 0:
            raise RuntimeError("N must be > 0")
        if num_hash_functions <= 0:
            raise RuntimeError("Must have at least 1 hash functions.")

        _full_table_size = int(N * 1.25) + 1
        _hash_type = np.int64
        self._max_size_chaining = int(7 * (log(N) / log(2)))
        _stash_size = (stash_size ** 2) + 1  # 101

        self._cuckoo_values = np.repeat(KEY_EMPTY, _full_table_size)
        self._stash_values = np.repeat(KEY_EMPTY, _stash_size)

        self._hash_functions = self._get_all_hash_tables(_full_table_size, num_hash_functions)
        self._stash_hash_function = self._get_stash_function(_stash_size)

        self._allow_parallel = parallel

    def get_multiple(self, keys: List[int]):
        @jit(nogil = True, cache = True)
        def helper(thread_index):
            key = keys[thread_index]
            return self._get_helper(key)

        return self._run_parallel(helper = helper,
                                  num_threads = len(keys))

    def get_single(self, key: int):
        return self._get_helper(key)

    def set(self, key: Union[int, List[int]], value: Union[np.uint64, List[np.uint64]]):
        if type(key) == int:
            return self.set_single(key, value)
        elif type(key) == list:
            return self.set_multiple(key, value)
        else:
            raise TypeError("Key must be int or List[int]")

    def set_single(self, key: int, value: np.uint64):
        return self.set_multiple(keys = [key], values = [value])

    def set_multiple(self, keys: List[int], values: List[np.uint64]):
        # These would be all parallel if CUDA implementation

        @jit(nogil = True, cache = True)
        def helper(thread_index):
            key = keys[thread_index]
            value = values[thread_index]
            did_insert = self._set_helper(key, value)
            return did_insert

        return self._run_parallel(helper = helper,
                                  num_threads = len(keys))

    def _run_parallel(self, helper, num_threads):
        if self._allow_parallel:
            thread_ids = prange(num_threads)
        else:
            thread_ids = range(num_threads)

        output = []
        for i in thread_ids:
            output.append(helper(i))
        return output

    def get(self, key: int):
        if type(key) == int:
            return self.get_single(key)
        elif type(key) == list:
            return self.get_multiple(key)
        else:
            raise TypeError("Key must be int or List[int]")

    @jit(nogil = True, cache = True)
    def _get_helper(self, key: int):
        locations = self._get_all_locations(key)
        entry = self._cuckoo_values[locations[0]]

        for location in locations:
            entry = self._cuckoo_values[location]
            if self._get_key(entry) == key:
                break
            elif self._get_key(entry) in KEY_EMPTY:
                return ENTRY_NOT_FOUND

        return self._get_value(entry)

    @jit(nogil = True, cache = True)
    def _set_helper(self, key: int, value: np.uint64) -> bool:
        entry = np.uint64(key << 32) + np.uint64(value)

        location_var = 0
        location = self._hash_functions[location_var](key)
        for its in range(self._max_size_chaining):
            entry, self._cuckoo_values[location] = self._cuckoo_values[location], entry
            key = self._get_key(entry)

            if key in KEY_EMPTY:
                return True

            locations = self._get_all_locations(key)
            location_var = (location_var + 1) % len(self._hash_functions)

            location = locations[location_var]

        # print("Resorting to slots")
        # We didn't find an empty slot, so we need to check the stash.
        slot = self._stash_hash_function(key)
        entry, self._stash_values[slot] = self._stash_values[slot], entry
        return entry in SLOT_EMPTY

    # Helper Functions
    def _get_key(self, entry: np.uint64):
        return np.uint64(entry >> 32)

    def _get_value(self, entry: int):
        return np.uint64(entry & 0xffffffff)

    def _get_all_locations(self, key: int):
        return list(map(lambda f: f(key), self._hash_functions))

    def _get_all_hash_tables(self, table_size: int, num_hash_functions: int):
        p = 4_294_967_291
        a = np.random.randint(low = 1, high = p, size = num_hash_functions)
        b = np.random.randint(low = 0, high = p, size = num_hash_functions)

        hash_functions = list(map(lambda i: self._get_next_hash_function(
                a = a[i],
                b = b[i],
                p = p,
                s_t = table_size,
                function = "xor"
        ), range(a.size)))

        return hash_functions

    def _get_stash_function(self, stash_size: int):
        p = 334_214_459
        return self._get_next_hash_function(a = random.randint(a = 1, b = p),
                                            b = random.randint(a = 0, b = p),
                                            p = p,
                                            s_t = stash_size,
                                            function = "linear")

    def _get_next_hash_function(self, a, b, p, s_t, function: str):
        def linear(k):
            return np.uint64((((a * k) + b) % p) % s_t)

        def xor(k):
            return np.uint64((((int(a) ^ int(k)) + b) % p) % s_t)

        if function == "linear":
            return linear
        elif function == "xor":
            return xor
        else:
            raise NotImplementedError("Invalid hash function: " + str(function))
