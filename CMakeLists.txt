cmake_minimum_required(VERSION 3.8)

project(CuckooStash CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_DIR "src")
include_directories(${SOURCE_DIR})
#set(SOURCES "${SOURCE_DIR}/my_math.cpp")

option(BUILD_PYTHON "Build Python SWIG module" ON)

include(cmake/pybind11.cmake)
fetch_pybind(
        ${PROJECT_SOURCE_DIR}/cmake
        ${PROJECT_BINARY_DIR}/pybind
)

#find_package(OpenCV)

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

find_package(Boost 1.62.0 COMPONENTS filesystem REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS} STATIC)
#    add_executable(${PROJECT_NAME} ${SOURCE_FILES})
endif()

if (BUILD_PYTHON)
    pybind11_add_module(example_wrapper ${SOURCE_DIR}/example_wrapper/my_math.cpp "${SOURCE_DIR}/example_wrapper/bindings.cpp")
    pybind11_add_module(cuckoo ${SOURCE_DIR}/CuckooCPU/cuckoo.cpp "${SOURCE_DIR}/CuckooCPU/bindings.cpp")
else()
    add_executable(cuckoo ${SOURCE_DIR}/CuckooCPU/cuckoo.cpp)
endif()

target_link_libraries(cuckoo PUBLIC ${Boost_LIBRARIES})

# Setup testing
# include(cmake/googletest.cmake)
#fetch_googletest(
#        ${PROJECT_SOURCE_DIR}/cmake
#        ${PROJECT_BINARY_DIR}/googletest
#)
#enable_testing()
#add_subdirectory(tests)
