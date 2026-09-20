#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <limits>
#include "Tracker.hpp"
#include "ByteTracker.hpp"
#include "Detection.hpp"
#include "Prediction.hpp"

namespace py = pybind11;

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(tracker, m)
{
    m.doc() = "Tracker library";
    m.def("add", &add, "A function that adds two numbers");

    py::class_<Detection>(m, "Detection")
        .def(py::init<double, double, double, double, double, int>(), py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"), py::arg("condifence"), py::arg("class_id"))
        .def_property_readonly("x", &Detection::x)
        .def_property_readonly("y", &Detection::y)
        .def_property_readonly("width", &Detection::width)
        .def_property_readonly("height", &Detection::height)
        .def_property_readonly("confince", &Detection::confidence)
        .def_property_readonly("class_id", &Detection::classId);

    py::class_<Prediction>(m, "Prediction").def_property_readonly("id", &Prediction::id).def_property_readonly("box", &Prediction::box, py::return_value_policy::copy);

    py::class_<ByteTrackerConfig>(m, "ByteTrackerConfig")
        .def(py::init<>())
        .def_readwrite("lowConfidence", &ByteTrackerConfig::lowConfidence)
        .def_readwrite("highConfidence", &ByteTrackerConfig::highConfidence)
        .def_readwrite("newTrackConfidence", &ByteTrackerConfig::newTrackConfidence)
        .def_readwrite("firstMatchCost", &ByteTrackerConfig::firstMatchCost)
        .def_readwrite("secondMatchCost", &ByteTrackerConfig::secondMatchCost)
        .def_readwrite("tentativeMatchCost", &ByteTrackerConfig::tentativeMatchCost)
        .def_readwrite("maxLostFrames", &ByteTrackerConfig::maxLostFrames);

    py::class_<Tracker>(m, "Tracker")
        .def("update", [](Tracker &tracker, py::array_t<float, py::array::c_style | py::array::forcecast> input)
             {
            const auto buf = input.request();
            if (buf.ndim != 2 || buf.shape[1] != 6)
                throw py::value_error("Expected numpy array with shape (N, 6)");

            const auto n = buf.shape[0];
            const auto *ptr = static_cast<const float *>(buf.ptr);
            std::vector<Detection> detections;
            detections.reserve(static_cast<std::size_t>(n));
            for (py::ssize_t i = 0; i < n; ++i)
            {
                const float *row = ptr + i * 6;
                const double classId = row[5];
                if (!std::isfinite(classId) || std::trunc(classId) != classId ||
                    classId < std::numeric_limits<int>::min() ||
                    classId > std::numeric_limits<int>::max())
                    throw py::value_error("class_id must be a finite integer in the C++ int range (-2147483648 to 2147483647)");
                detections.emplace_back(row[0], row[1], row[2], row[3], row[4],
                                        static_cast<int>(classId));
            }

            const auto tracks = tracker.update(detections);
            py::array_t<float> output({static_cast<py::ssize_t>(tracks.size()), py::ssize_t{7}});
            auto out = output.mutable_unchecked<2>();
            for (std::size_t i = 0; i < tracks.size(); ++i)
            {
                const auto &box = tracks[i].box();
                out(i, 0) = static_cast<float>(box.x());
                out(i, 1) = static_cast<float>(box.y());
                out(i, 2) = static_cast<float>(box.width());
                out(i, 3) = static_cast<float>(box.height());
                out(i, 4) = static_cast<float>(tracks[i].id());
                out(i, 5) = static_cast<float>(box.confidence());
                out(i, 6) = static_cast<float>(box.classId());
            }
            return output; }, py::arg("detections"), "Update one frame: (N, 6) [x, y, width, height, confidence, class_id] "
                                         "to float32 (N, 7) [x, y, width, height, track_id, confidence, class_id], "
                                         "where N is the number of returned tracks.")
        .def("reset", &Tracker::reset);

    py::class_<ByteTracker, Tracker>(m, "ByteTracker")
        .def(py::init<ByteTrackerConfig>(),
             py::arg_v("config", ByteTrackerConfig{}, "ByteTrackerConfig()"));
}
