#pragma once

#include "BoxFormat.hpp"
#include "Detection.hpp"
#include "Prediction.hpp"
#include <vector>
#include <stdexcept>

class Tracker
{
public:
    virtual ~Tracker() = default;
    // Raw input format; Detection objects and predictions always use TLWH.
    virtual BoxFormat inputFormat() const { return _inputFormat; }
    // One call per frame, including frames with no detections.
    virtual std::vector<Prediction> update(const std::vector<Detection> &detections) = 0;
    virtual void reset() = 0;

protected:
    explicit Tracker(BoxFormat inputFormat = BoxFormat::TLWH) : _inputFormat(inputFormat)
    {
        switch (inputFormat)
        {
        case BoxFormat::TLWH:
        case BoxFormat::CXCYWH:
        case BoxFormat::XYXY:
            break;
        default:
            throw std::invalid_argument("Invalid input box format");
        }
    }

private:
    BoxFormat _inputFormat;
};
