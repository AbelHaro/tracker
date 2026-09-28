#pragma once
#include "Tracker.hpp"
#include <pybind11/numpy.h>

namespace tracker_bindings
{
    namespace py = pybind11;
    using DetectionArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
    std::vector<Detection> readDetections(const DetectionArray &input, BoxFormat format);
    py::array_t<float> writePredictions(const std::vector<Prediction> &tracks);
    py::array_t<float> update(Tracker &tracker, const DetectionArray &input);
}
