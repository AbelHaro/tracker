#pragma once

#include "Detection.hpp"
#include "Prediction.hpp"
#include <vector>

class Tracker
{
public:
    virtual ~Tracker() = default;
    // One call per frame, including frames with no detections.
    virtual std::vector<Prediction> update(const std::vector<Detection> &detections) = 0;
    virtual void reset() = 0;
};
