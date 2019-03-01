FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "")

# Include python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

include_directories(../src)

# Add swig module
SET_SOURCE_FILES_PROPERTIES(../src/cuckoo.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(../src/cuckoo.i PROPERTIES SWIG_FLAGS "-includeall")

swig_add_library(Cuckoo LANGUAGE python SOURCES ../src/cuckoo.i ../src/cuckoo.cpp)
swig_link_libraries(Cuckoo ${PYTHON_LIBRARIES})

# Files to install with Python
set(PYTHON_INSTALL_FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Cuckoo.py
        ${CMAKE_CURRENT_BINARY_DIR}/_Cuckoo.so)

# Configure setup.py and copy to output directory
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in.py)
set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

# Declare install target for python
#install(TARGETS Cuckoo
#        COMMAND "${PYTHON_EXECUTABLE} setup.py"
#        COMPONENT Cuckoo_lib)

# Install target to call setup.py
add_custom_target(build-python
        DEPENDS Cuckoo
        COMMAND python ${SETUP_PY_OUT} build)