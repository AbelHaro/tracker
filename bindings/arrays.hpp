#pragma once
#include "Tracker.hpp"
#include <pybind11/numpy.h>

namespace tracker_bindings
{
    namespace py = pybind11;
    std::vector<Detection> readDetections(const py::array_t<float> &input, BoxFormat format);
    std::vector<Prediction> update(Tracker &tracker, const py::array_t<float> &input);
}
