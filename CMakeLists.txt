cmake_minimum_required(VERSION 3.7)

project(CuckooStash CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_DIR "src")
include_directories(${SOURCE_DIR})
#set(SOURCES "${SOURCE_DIR}/my_math.cpp")

#option(BUILD_PYTHON "Build Python SWIG module" ON)

#if (BUILD_PYTHON)
#    add_subdirectory(python)
#endif()

include(cmake/pybind11.cmake)
fetch_pybind(
        ${PROJECT_SOURCE_DIR}/cmake
        ${PROJECT_BINARY_DIR}/pybind
)

pybind11_add_module(example_wrapper ${SOURCE_DIR}/example_wrapper/my_math.cpp "${SOURCE_DIR}/example_wrapper/bindings.cpp")
pybind11_add_module(cuckoo ${SOURCE_DIR}/CuckooCPU/cuckoo.cpp "${SOURCE_DIR}/CuckooCPU/bindings.cpp")

# Setup testing
# include(cmake/googletest.cmake)
#fetch_googletest(
#        ${PROJECT_SOURCE_DIR}/cmake
#        ${PROJECT_BINARY_DIR}/googletest
#)
#enable_testing()
#add_subdirectory(tests)
