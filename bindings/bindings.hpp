#pragma once
#include <pybind11/pybind11.h>

namespace tracker_bindings
{
    namespace py = pybind11;
    void bindCommon(py::module_ &m);
    void bindSort(py::module_ &m);
    void bindByteTracker(py::module_ &m);
}
