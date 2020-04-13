cmake_minimum_required(VERSION 3.8)
project(CuckooStash LANGUAGES CXX)

# set build type to release by default
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build, options are: Debug Release." FORCE)
endif()
if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Release" AND NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be set to Release or Debug")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (APPLE)
   include(/Users/matthew/coriander/share/cocl/cocl.cmake)
endif()

option(CMAKE_USE_WIN32_THREADS_INIT "using WIN32 threads" ON)

option(ENABLE_COVERAGE "Enable coverage" OFF)
option(ENABLE_ALL_SAN "Enable All Sanitizers" OFF)
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(ENABLE_UBSAN "Enable Undefined Behaviour Sanitizer" OFF)
option(ENABLE_TSAN "Enable Thread Sanitizer" OFF)
option(ENABLE_LSAN "Enable Leak Sanitizer" OFF)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	message("Memory and refcounting: ENABLED")
  option(ENABLE_REF_DEBUG "enable memory and refcounting debugging" ON)
else()
	message("Memory and refcounting: DISABLED")
  option(ENABLE_REF_DEBUG "enable memory and refcounting debugging" OFF)
endif()

option(ENABLE_CCACHE "enable caching compiler output" OFF)
option(ENABLE_GOLD "enable the gold linker" OFF)
option(ENABLE_GPERFTOOLS "enable the google performance tools" OFF)
option(ENABLE_PGO "enable -fprofile-generate/-fprofile-use" OFF)
option(ENABLE_LTO "enable -flto" OFF)

option(BUILD_PYTHON "Build Python pybind module" OFF)

# include(cmake/pybind11.cmake)
#fetch_pybind(
#        ${PROJECT_SOURCE_DIR}/cmake
#        ${PROJECT_BINARY_DIR}/pybind
#)

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE true)
# Used to enable debugging (gdb)
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    add_compile_options("-g")
endif()

## These two things are needed for YCM
SET( CMAKE_EXPORT_COMPILE_COMMANDS ON )
IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()

if (ENABLE_ASAN OR ENABLE_ALL_SAN)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

if (ENABLE_UBSAN OR ENABLE_ALL_SAN)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
endif()

if (ENABLE_TSAN OR ENABLE_ALL_SAN)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
endif()

if (ENABLE_LSAN OR ENABLE_ALL_SAN)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=leak")
endif()

#set(Boost_USE_MULTITHREADED OFF)
#find_package(Boost REQUIRED COMPONENTS system)

#if (ENABLE_COVERAGE)
#    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Set the build type." FORCE)
#    include(cmake/CodeCoverage.cmake)
#    append_coverage_compiler_flags()
#endif()

#find_package(Catch2 REQUIRED)

#if (ENABLE_COVERAGE)
#       setup_target_for_coverage_lcov(
#          NAME coverage_lispp
#          EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/test_lispp
#          LCOV_ARGS --rc lcov_branch_coverage=1
#          GENHTML_ARGS --legend --branch-coverage
#          DEPENDENCIES lispp)
#endif()

# automatically use ccache if found
if(ENABLE_CCACHE AND NOT ENABLE_PGO)
  find_program(CCACHE ccache)
  if(CCACHE)
    message(STATUS "found ccache ${CCACHE}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "CCACHE_CPP2=yes ${CCACHE}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE})
  endif()
endif()

if(ENABLE_LTO)
  set(LTO_FLAGS "-flto")
endif()

if(ENABLE_PGO)
  if(NOT "${PROFILE_DIR}" STREQUAL "")
    set(PROFILE_STATE "${PROFILE_STATE}=${PROFILE_DIR}")
  endif()
  set(PGO_FLAGS "-fprofile-${PROFILE_STATE} -fprofile-correction")
endif()

if(ENABLE_REF_DEBUG)
 add_definitions(-DPy_REF_DEBUG)
 add_definitions(-DPYMALLOC_DEBUG)
 add_definitions(-DPy_TRACE_REFS)
endif()

# automatically use the gold linker if found
if(ENABLE_GOLD)
  find_program(GOLD_LINKER ld.gold)
  if(GOLD_LINKER)
    message(STATUS "found the gold linker ${GOLD_LINKER}")
    set(CMAKE_LINKER "${GOLD_LINKER}")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -B${CMAKE_SOURCE_DIR}/tools/build_system")
    if(NOT ENABLE_PGO)
      # let pgo determine the ordering
      #set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -Wl,--section-ordering-file,${CMAKE_CURRENT_SOURCE_DIR}/section_ordering.txt")
    endif()
  endif()
endif()

# valgrind
if(ENABLE_VALGRIND)
  find_package(Valgrind REQUIRED)
  include_directories(${VALGRIND_INCLUDE_DIR})
  message(STATUS "Including valgrind ${VALGRIND_INCLUDE_DIR}")
else()
  add_definitions(-DNVALGRIND)
endif()


if(ENABLE_GPERFTOOLS)
  set(OPTIONAL_LIBRARIES ${OPTIONAL_LIBRARIES} profiler)
endif()


include_directories(.)
if (APPLE)
    add_library(cuckoo cuckoo/cpu/cuckoo.cpp)
else()
	find_package(Threads REQUIRED)
	find_package(CUDA)
    cuda_add_library(cuckoo cuckoo/gpu/array.hpp cuckoo/gpu/cuckoo.hpp cuckoo/gpu/cuckoo.cu)
    # target_link_libraries(cuckoo pybind11::module)
    target_link_libraries(cuckoo Threads::Threads)
endif()
# set_target_properties(cuckoolib PROPERTIES LINKER_LANGUAGE CXX)

find_package(Catch2 CONFIG REQUIRED)

add_executable(test_cuckoo app/test_cuckoo.cpp)
target_link_libraries(test_cuckoo cuckoo Catch2::Catch2)

list(APPEND perf_targets test_cuckoo)
#if (BUILD_PYTHON)
#    add_library(cuckoo src/CuckooGPU/cuckoo.hpp)
#	target_link_libraries(cuckoo pybind11::module cuckoolib)
	#else()
	#add_library(cuckoo src/CuckooCPU/cuckoo.mm)
	#configure_file(src/CuckooCPU/MyKernels.metallib ${CMAKE_BINARY_DIR}/MyKernels.metallib COPYONLY)
	#endif()

#if (APPLE)
#    target_compile_definitions(cuckoo PRIVATE CUCKOO_SUPPORT_METAL=1)

#    find_library(METAL_LIBRARY Metal)
#    find_library(METALKIT_LIBRARY MetalKit)
#    find_library(COCOA_LIBRARY Cocoa)
#    find_library(CORE_FOUNDATION_LIBARY Foundation)

#    set(CMAKE_CXX_FLAGS "-x objective-c++ -fobjc-link-runtime")
#    xcrun metal -fcikernel src/CuckooCPU/example.metal -c -o src/CuckooCPU/MyLibrary.air
#    xcrun metallib -cikernel src/CuckooCPU/MyLibrary.air -o src/CuckooCPU/MyKernels.metallib

#    add_compile_options(cuckoo "-mmacosx-version-min=10.13,-fPIC,-ObjC++")
#    add_link_options(cuckoo "-fobjc-link-runtime,-mmacosx-version-min=10.13,-fPIC")
#
#    SET(EXTRA_LIBS "${METAL_LIBRARY}" "${METALKIT_LIBRARY}" "${COCOA_LIBRARY}" "${CORE_FOUNDATION_LIBARY}")
#    message(${EXTRA_LIBS})
#else()
#    set(EXTRA_LIBS "")
#endif()

# Setup testing
# include(cmake/googletest.cmake)
#fetch_googletest(
#        ${PROJECT_SOURCE_DIR}/cmake
#        ${PROJECT_BINARY_DIR}/googletest
#)
#enable_testing()
#add_subdirectory(tests)
# doxygen

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(docs ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else()
  add_custom_target(docs COMMAND ${CMAKE_COMMAND} -E echo "Can't create docs, doxygen not installed \(try sudo apt-get install doxygen grpahviz on Ubuntu and then rerun cmake\)" VERBATIM)
endif()


# {run,dbg,perf,memcheck,memleaks,cachegrind}_TESTNAME
foreach(RUNTARGET ${perf_targets})
  get_filename_component(BASEFILENAME ${RUNTARGET} NAME_WE)
  add_custom_target(run_${BASEFILENAME} ./pyston -q ${RUNTARGET} DEPENDS pyston)
  add_custom_target(dbg_${BASEFILENAME} gdb --ex "set confirm off" --ex "handle SIGUSR2 pass nostop noprint" --ex run --ex "bt 20" --args ./pyston -q ${RUNTARGET} DEPENDS pyston)
  add_custom_target(perf_${BASEFILENAME} perf record -g -- ./pyston -q -p ${RUNTARGET}
                    COMMAND perf report -v -n -g flat,1000 | bash ${CMAKE_SOURCE_DIR}/tools/cumulate.sh | less -S)

  if(ENABLE_VALGRIND)
    add_custom_target(memcheck_${BASEFILENAME} valgrind --tool=memcheck --leak-check=no --db-attach=yes ./pyston ${RUNTARGET} DEPENDS pyston)
    add_custom_target(memleaks_${BASEFILENAME} valgrind --tool=memcheck --leak-check=full --leak-resolution=low --show-reachable=yes ./pyston ${RUNTARGET} DEPENDS pyston)
    add_custom_target(cachegrind_${BASEFILENAME} valgrind --tool=cachegrind ./pyston ${RUNTARGET} DEPENDS pyston)
  endif()
endforeach()
