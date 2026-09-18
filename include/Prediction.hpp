#pragma once

#include "Detection.hpp"
#include <cstdint>

// Output snapshot. Identity belongs to the track, never to a prediction.
class Prediction
{
public:
    Prediction(std::uint64_t id, const Detection &box) : _id(id), _box(box) {}
    std::uint64_t id() const { return _id; }
    const Detection &box() const { return _box; }

private:
    std::uint64_t _id;
    Detection _box;
};
