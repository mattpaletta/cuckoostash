macro(fetch_pybind _download_module_path _download_root)
    set(PYBIND_DOWNLOAD_ROOT ${_download_root})
    configure_file(
            ${_download_module_path}/pybind11-download.cmake
            ${_download_root}/CMakeLists.txt
            @ONLY
    )
    unset(PYBIND_DOWNLOAD_ROOT)

    execute_process(
            COMMAND
            "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY
            ${_download_root}
    )
    execute_process(
            COMMAND
            "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY
            ${_download_root}
    )

    add_subdirectory(
            ${_download_root}/pybind-src
            ${_download_root}/pybind-build
    )
endmacro()

macro(fetch_catch2 _download_module_path _download_root)
    set(PYBIND_DOWNLOAD_ROOT ${_download_root})
    configure_file(
            ${_download_module_path}/pybind11-download.cmake
            ${_download_root}/CMakeLists.txt
            @ONLY
    )
    unset(PYBIND_DOWNLOAD_ROOT)

    execute_process(
            COMMAND
            "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY
            ${_download_root}
    )
    execute_process(
            COMMAND
            "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY
            ${_download_root}
    )

    add_subdirectory(
            ${_download_root}/catch2-src
            ${_download_root}/catch2-build
    )
endmacro()
