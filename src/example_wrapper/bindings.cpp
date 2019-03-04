//
// Created by Matthew Paletta on 2019-03-03.
//

#include <pybind11/pybind11.h>
#include "my_math.h"

namespace py = pybind11;

PYBIND11_PLUGIN(example_wrapper) {
                py::module m("example_wrapper");
                m.def("add", &add);
                m.def("subtract", &subtract);
                return m.ptr();
}