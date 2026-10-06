#include "bindings.hpp"
#include "arrays.hpp"
#include <pybind11/stl.h>

namespace tracker_bindings
{
    void bindCommon(py::module_ &m)
    {
        py::enum_<BoxFormat>(m, "BoxFormat")
            .value("TLWH", BoxFormat::TLWH)
            .value("CXCYWH", BoxFormat::CXCYWH)
            .value("XYXY", BoxFormat::XYXY);

        py::class_<Detection>(m, "Detection")
            .def(py::init<double, double, double, double, double, int>(), py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"), py::arg("confidence"), py::arg("class_id"))
            .def_property_readonly("x", &Detection::x)
            .def_property_readonly("y", &Detection::y)
            .def_property_readonly("width", &Detection::width)
            .def_property_readonly("height", &Detection::height)
            .def_property_readonly("confidence", &Detection::confidence)
            .def_property_readonly("class_id", &Detection::classId);

        py::class_<Prediction>(m, "Prediction", "An output snapshot containing a track ID and its TLWH detection box.")
            .def_property_readonly("id", &Prediction::id)
            .def_property_readonly("box", &Prediction::box, py::return_value_policy::copy);

        py::class_<Tracker>(m, "Tracker")
            .def_property_readonly("input_format", &Tracker::inputFormat)
            .def("update", &update, py::arg("detections").noconvert(), "Update one frame with a float32 NumPy array: (N, 6) [a, b, c, d, confidence, class_id]. "
                                                           "Pixel box coordinates follow input_format: TLWH (x, y, width, height), "
                                                           "CXCYWH (center_x, center_y, width, height), or XYXY (x1, y1, x2, y2). "
                                                           "Returns a list of Prediction snapshots with integer id and box properties "
                                                           "with top-left x, y (TLWH), regardless of input_format. Empty frames use shape (0, 6).")
            .def("reset", &Tracker::reset);
    }
}
