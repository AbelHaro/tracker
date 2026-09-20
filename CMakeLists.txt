cmake_minimum_required(VERSION 3.20...4.2)
project(tracker_cpp LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 CONFIG REQUIRED)

pybind11_add_module(tracker
    bindings/tracker.cpp
    src/AssociationCost.cpp
    src/ByteTracker.cpp
    src/HungarianAlgorithm.cpp
    src/Track.cpp
)
target_compile_features(tracker PRIVATE cxx_std_23)
set_target_properties(tracker PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(tracker PRIVATE include)
target_link_libraries(tracker PRIVATE Eigen3::Eigen)

install(TARGETS tracker LIBRARY DESTINATION . RUNTIME DESTINATION .)
install(FILES bindings/tracker.pyi DESTINATION .)
