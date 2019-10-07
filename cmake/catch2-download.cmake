cmake_minimum_required(VERSION 3.7)

project(pybind11-download NONE)

include(ExternalProject)

ExternalProject_Add(
        pybind11
        SOURCE_DIR "@PYBIND_DOWNLOAD_ROOT@/pybind-src"
        BINARY_DIR "@PYBIND_DOWNLOAD_ROOT@/pybind-build"
        GIT_REPOSITORY
        https://github.com/pybind/pybind11.git
        GIT_TAG
        v2.2.4
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
)