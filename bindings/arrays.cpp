#include "arrays.hpp"
#include <cmath>
#include <limits>

namespace tracker_bindings
{
    std::vector<Detection> readDetections(const py::array_t<float> &input, BoxFormat format)
    {
        const auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[1] != 6)
            throw py::value_error("Expected numpy array with shape (N, 6)");

        const auto n = buf.shape[0];
        const auto values = input.unchecked<2>();
        std::vector<Detection> detections;
        detections.reserve(static_cast<std::size_t>(n));

        for (py::ssize_t i = 0; i < n; ++i)
        {
            const double classId = values(i, 5);
            if (!std::isfinite(classId) || std::trunc(classId) != classId ||
                classId < std::numeric_limits<int>::min() ||
                classId > std::numeric_limits<int>::max())
                throw py::value_error("class_id must be a finite integer in the C++ int range (-2147483648 to 2147483647)");
            double x = values(i, 0), y = values(i, 1), width = values(i, 2), height = values(i, 3);
            switch (format)
            {
            case BoxFormat::TLWH:
                break;
            case BoxFormat::CXCYWH:
                x -= width / 2;
                y -= height / 2;
                break;
            case BoxFormat::XYXY:
                width -= x;
                height -= y;
                break;
            default:
                throw py::value_error("Invalid input box format");
            }
            detections.emplace_back(x, y, width, height, values(i, 4),
                                    static_cast<int>(classId));
        }
        return detections;
    }

    std::vector<Prediction> update(Tracker &tracker, const py::array_t<float> &input)
    {
        return tracker.update(readDetections(input, tracker.inputFormat()));
    }
}
