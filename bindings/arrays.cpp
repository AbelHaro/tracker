#include "arrays.hpp"
#include <cmath>
#include <limits>

namespace tracker_bindings
{
    std::vector<Detection> readDetections(const DetectionArray &input, BoxFormat format)
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
            double x = row[0], y = row[1], width = row[2], height = row[3];
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
            detections.emplace_back(x, y, width, height, row[4],
                                    static_cast<int>(classId));
        }
        return detections;
    }

    py::array_t<float> writePredictions(const std::vector<Prediction> &tracks)
    {
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
        return output;
    }

    py::array_t<float> update(Tracker &tracker, const DetectionArray &input)
    {
        return writePredictions(tracker.update(readDetections(input, tracker.inputFormat())));
    }
}
