cmake_minimum_required(VERSION 3.8)

project(CuckooStash CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_PYTHON "Build Python SWIG module" OFF)

include(cmake/pybind11.cmake)
fetch_pybind(
        ${PROJECT_SOURCE_DIR}/cmake
        ${PROJECT_BINARY_DIR}/pybind
)

#find_package(OpenCV)

#if (OpenCV_FOUND)
#    include_directories(${OpenCV_INCLUDE_DIRS})
#endif()

#find_package(Boost 1.62.0 COMPONENTS filesystem REQUIRED)
#if(Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIRS} STATIC)
##    add_executable(${PROJECT_NAME} ${SOURCE_FILES})
#endif()

if (BUILD_PYTHON)
#    pybind11_add_module(example_wrapper src/example_wrapper/my_math.cpp "src/example_wrapper/bindings.cpp")
    pybind11_add_module(cuckoo src/CuckooCPU/bindings.mm)
else()
    add_library(cuckoo src/CuckooCPU/cuckoo.mm)
    configure_file(src/CuckooCPU/MyKernels.metallib ${CMAKE_BINARY_DIR}/MyKernels.metallib COPYONLY)
endif()

if (APPLE)
    target_compile_definitions(cuckoo PRIVATE CUCKOO_SUPPORT_METAL=1)

    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    find_library(COCOA_LIBRARY Cocoa)
    find_library(CORE_FOUNDATION_LIBARY Foundation)

#    set(CMAKE_CXX_FLAGS "-x objective-c++ -fobjc-link-runtime")

    add_compile_options(cuckoo "-mmacosx-version-min=10.13,-fPIC,-ObjC++")
    add_link_options(cuckoo "-fobjc-link-runtime,-mmacosx-version-min=10.13,-fPIC")

    SET(EXTRA_LIBS "${METAL_LIBRARY}" "${METALKIT_LIBRARY}" "${COCOA_LIBRARY}" "${CORE_FOUNDATION_LIBARY}")
    message(${EXTRA_LIBS})
else()
    set(EXTRA_LIBS "")
endif()

target_link_libraries(cuckoo PUBLIC ${EXTRA_LIBS})

# Setup testing
# include(cmake/googletest.cmake)
#fetch_googletest(
#        ${PROJECT_SOURCE_DIR}/cmake
#        ${PROJECT_BINARY_DIR}/googletest
#)
#enable_testing()
#add_subdirectory(tests)
