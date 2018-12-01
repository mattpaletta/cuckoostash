import random
from math import log
from os import cpu_count
from time import time
from typing import List, Union

import numpy as np
from multiprocess.pool import Pool

ENTRY_NOT_FOUND = KEY_EMPTY = SLOT_EMPTY = 0xffff



g_pool = Pool(processes = cpu_count() - 1)

class CuckooCpu(object):
    def __init__(self, N: int, stash_size = 10, num_hash_functions = 4, num_parallel = cpu_count() - 1):
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

        global g_pool
        g_pool = Pool(processes = num_parallel)

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

    def get_multiple(self, keys: List[int]):
        return g_pool.map(self.get, keys, chunksize = 10)

    def get(self, key: int):
        locations = self._get_all_locations(key)
        entry = self._cuckoo_values[locations[0]]

        for location in locations:
            entry = self._cuckoo_values[location]
            if self._get_key(entry) != key:
                if self._get_key(entry) == KEY_EMPTY:
                    return ENTRY_NOT_FOUND
            else:
                break

        return self._get_value(entry)

    def _get_key(self, entry: int):
        return np.uint64(entry >> 32)

    def _get_value(self, entry: int):
        return np.uint64(entry & 0xffffffff)

    def _get_all_locations(self, key: int):
        return list(map(lambda f: f(key), self._hash_functions))

    def set(self, key: Union[int, List[int]], value: Union[np.uint64, List[np.uint64]]):
        if type(key) == int:
            return self.set_single(key, value)
        elif type(key) == list:
            return self.set_multiple(key, value)

    def set_single(self, key: int, value: np.uint64):
        return self.set_multiple(keys = [key], values = [value])

    def set_multiple(self, keys: List[int], values: List[np.uint64]):
        # These would be all parallel if CUDA implementation
        def helper(thread_index):
            key = keys[thread_index]
            value = values[thread_index]
            did_insert = self._set_helper(key, value)
            return did_insert
        return g_pool.map(helper, range(len(keys)), chunksize = 10)

    def _set_helper(self, key: int, value: np.uint64) -> bool:
        location = self._hash_functions[0](key)
        entry = np.uint64(key << 32) + np.uint64(value)

        for its in range(self._max_size_chaining):
            entry, self._cuckoo_values[location] = self._cuckoo_values[location], entry
            key = self._get_key(entry)

            if key == KEY_EMPTY:
                return True

            locations = list(map(lambda f: f(key), self._hash_functions))
            current_location = its % len(self._hash_functions)
            location = locations[current_location]

        # print("Resorting to slots")
        # We didn't find an empty slot, so we need to check the stash.
        slot = self._stash_hash_function(key)
        entry, self._stash_values[slot] = self._stash_values[slot], entry
        return entry == SLOT_EMPTY


if __name__ == "__main__":

    num_elements = 100_000

    cuckoo = CuckooCpu(N = int(num_elements * 3),
                       stash_size = 10,
                       num_hash_functions = 4)

    keys = list(range(num_elements + 1))
    values = list(map(lambda x: x + 100, keys))

    print("Running set")
    start = time()
    cuckoo.set(keys, values)
    end = time()
    print("Finished set")
    print(end - start)

    print("Running get")
    start1 = time()
    cuckoo.get_multiple(keys)
    end1 = time()
    print("Finished get")
    print(end1 - start1)