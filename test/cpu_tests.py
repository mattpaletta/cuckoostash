from unittest import TestCase

from cuckoo.cpu.cuckoo import CuckooCpu


class TestPQDict(TestCase):
    def setUp(self):
        self._cuckoo = CuckooCpu(N = 10)