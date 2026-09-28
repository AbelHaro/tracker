#include "bindings.hpp"
#include "ByteTracker.hpp"

namespace tracker_bindings
{
    void bindByteTracker(py::module_ &m)
    {
        py::class_<ByteTracker, Tracker>(m, "ByteTracker")
            .def(py::init<double, double, double, double, double, double, int, BoxFormat, int>(),
                 py::kw_only(),
                 py::arg("low_confidence") = 0.1,
                 py::arg("high_confidence") = 0.6,
                 py::arg("new_track_confidence") = 0.7,
                 py::arg("first_match_cost") = 0.8,
                 py::arg("second_match_cost") = 0.5,
                 py::arg("tentative_match_cost") = 0.7,
                 py::arg("max_lost_frames") = 30,
                 py::arg_v("input_format", BoxFormat::TLWH, "BoxFormat.TLWH"),
                 py::arg("threads") = 0);
    }
}
