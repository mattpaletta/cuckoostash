//
// Created by Matthew Paletta on 2019-03-03.
//

#include <pybind11/pybind11.h>
#include "cuckoo.h"

namespace py = pybind11;

PYBIND11_MODULE(cuckoo, m) {
    py::class_<Cuckoo> cuckoo(m, "Cuckoo");
            cuckoo.def(py::init<unsigned int, int, int>());
    cuckoo.def("set", &Cuckoo::set);
    cuckoo.def("get", &Cuckoo::get);
}