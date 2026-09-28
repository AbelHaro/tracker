#pragma once

#include "BoxFormat.hpp"
#include "Detection.hpp"
#include "Prediction.hpp"
#include <vector>

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
    explicit Tracker(BoxFormat inputFormat = BoxFormat::TLWH) : _inputFormat(inputFormat) {}

private:
    BoxFormat _inputFormat;
};
