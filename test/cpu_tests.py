from unittest import TestCase
import numpy as np

from cuckoo.cpu.cuckoo import CuckooCpu, ENTRY_NOT_FOUND

CAT = 1
DOG = 2
MOUSE = 3


class TestCPU(TestCase):
    def setUp(self):
        self._cuckoo = CuckooCpu(N = 4)

    def test_get(self):
        assert self._cuckoo.get(key = CAT) == ENTRY_NOT_FOUND, "Get on an empty list should be -1 (not found)"

    def test_set(self):
        self._cuckoo.set(key = CAT, value = np.uint64(1))
        assert self._cuckoo.get(key = CAT) == 1, "Get should return the last value entered."

    def test_set_single(self):
        self._cuckoo.set_single(key = CAT, value = np.uint64(1))
        assert self._cuckoo.get(key = CAT) == 1, "Get/Set should return the last value entered."

    def test_set_multiple(self):
        self._cuckoo.set(key = [CAT, DOG], value = [np.uint64(1), np.uint64(2)])
        assert self._cuckoo.get(key = CAT) == 1, "Get/Set should return the last value entered."
        assert self._cuckoo.get(key = DOG) == 2, "Get/Set should return the last value entered."

    def test_set_multiple2(self):
        self._cuckoo.set_multiple(keys = [CAT, DOG], values = [np.uint64(1), np.uint64(2)])
        assert self._cuckoo.get(key = CAT) == 1, "Get/Set should return the last value entered."
        assert self._cuckoo.get(key = DOG) == 2, "Get/Set should return the last value entered."

    def test_set_full(self):
        keys = [CAT, DOG, MOUSE]
        values = [np.uint64(1), np.uint64(2), np.uint64(3)]
        success = self._cuckoo.set_multiple(keys, values)

        for k, v, s in zip(keys, values, success):
            if s:
                assert self._cuckoo.get(k) == v, "All the successes should be there!"
