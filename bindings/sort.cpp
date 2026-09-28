#include "bindings.hpp"
#include "Sort.hpp"

namespace tracker_bindings
{
    void bindSort(py::module_ &m)
    {
        py::class_<Sort, Tracker>(m, "Sort")
            .def(py::init<double, int, BoxFormat>(), py::kw_only(),
                 py::arg("max_match_cost") = 0.7,
                 py::arg("max_lost_frames") = 30,
                 py::arg_v("input_format", BoxFormat::TLWH, "BoxFormat.TLWH"),
                 "Create SORT. max_match_cost limits 1 - IoU; 0.7 requires IoU >= 0.3.");
    }
}
