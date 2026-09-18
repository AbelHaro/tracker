#include "Track.hpp"

Track::Track(std::uint64_t id, const Detection &detection, bool confirmed)
    : _id(id), _lastDetection(detection),
      _state(confirmed ? TrackState::Tracked : TrackState::Tentative)
{
    _filter.initialize(detection.centerX(), detection.centerY());
}

void Track::predict()
{
    _filter.predict();
    ++_age;
    ++_missedFrames;
}

void Track::update(const Detection &detection)
{
    _filter.update(detection.centerX(), detection.centerY());
    _lastDetection = detection;
    _missedFrames = 0;
    _state = TrackState::Tracked;
}

Detection Track::box() const
{
    return Detection(_filter.x() - _lastDetection.width() / 2,
                     _filter.y() - _lastDetection.height() / 2,
                     _lastDetection.width(), _lastDetection.height(),
                     _lastDetection.confidence());
}
