#include "bindings.hpp"

PYBIND11_MODULE(tracker, m)
{
    m.doc() = "Object tracking with a shared NumPy interface.";
    m.def("add", [](int i, int j)
          { return i + j; }, "A function that adds two numbers");
    tracker_bindings::bindCommon(m);
    tracker_bindings::bindSort(m);
    tracker_bindings::bindByteTracker(m);
}
