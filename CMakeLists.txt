cmake_minimum_required(VERSION 3.20...4.2)
project(tracker_cpp LANGUAGES CXX)

include(CTest)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(tracker_core STATIC
    src/AssociationCost.cpp
    src/ByteTracker.cpp
    src/HungarianAlgorithm.cpp
    src/Track.cpp
)
target_compile_features(tracker_core PUBLIC cxx_std_23)
set_target_properties(tracker_core PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_include_directories(tracker_core PUBLIC include)
target_link_libraries(tracker_core PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

pybind11_add_module(tracker bindings/tracker.cpp)
target_link_libraries(tracker PRIVATE tracker_core)
set_target_properties(tracker PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)
add_custom_target(bindings DEPENDS tracker)

add_executable(tracker_example src/main.cpp)
target_link_libraries(tracker_example PRIVATE tracker_core)
set_target_properties(tracker_example PROPERTIES
    OUTPUT_NAME tracker
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
add_custom_target(run
    COMMAND $<TARGET_FILE:tracker_example>
    DEPENDS tracker_example
    USES_TERMINAL
)

if(BUILD_TESTING)
    foreach(test_name IN ITEMS kalman_filter_test tracker_test)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE tracker_core)
        set_target_properties(${test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

add_executable(association_benchmark EXCLUDE_FROM_ALL tests/association_benchmark.cpp)
target_link_libraries(association_benchmark PRIVATE tracker_core)
set_target_properties(association_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
add_custom_target(benchmark
    COMMAND $<TARGET_FILE:association_benchmark>
    DEPENDS association_benchmark
    USES_TERMINAL
)

find_program(UV_EXECUTABLE NAMES uv)
if(UV_EXECUTABLE)
    add_custom_target(stubs
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:tracker>"
            ${UV_EXECUTABLE} run --no-editable pybind11-stubgen --exit-code
            -o ${CMAKE_CURRENT_SOURCE_DIR}/bindings tracker
        # COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bindings/annotate_stub.py
        DEPENDS tracker
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL
    )
endif()

install(TARGETS tracker LIBRARY DESTINATION . RUNTIME DESTINATION .)
install(FILES bindings/tracker.pyi DESTINATION .)
