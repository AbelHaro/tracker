#pragma once

#include "KalmanFilter.hpp"
#include "Prediction.hpp"

// Internal trajectory state, independent of the association policy.
enum class TrackState
{
    Tentative,
    Tracked,
    Lost,
    Removed
};

class Track
{
public:
    Track(std::uint64_t id, const Detection &detection, bool confirmed);
    void predict();
    void update(const Detection &detection);
    void markLost() { _state = TrackState::Lost; }
    void markRemoved() { _state = TrackState::Removed; }
    TrackState state() const { return _state; }
    int missedFrames() const { return _missedFrames; }
    int age() const { return _age; }
    Detection box() const;
    Prediction prediction() const { return Prediction(_id, box()); }
    ~Track() = default;

private:
    std::uint64_t _id;
    KalmanFilter2D _filter;
    Detection _lastDetection;
    TrackState _state;
    int _age = 1;
    int _missedFrames = 0;
};
