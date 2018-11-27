cdef extern from "cuckoo.h" namespace "pycuckoo":
    cdef cppclass Cuckoo:
        Cuckoo()
        int set()

cdef class PyCuckoo:
    cdef Cuckoo *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new Cuckoo()

    def __dealloc__(self):
        del self.thisptr

    def set(self):
        return self.thisptr.set()