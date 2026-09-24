#include "ByteTracker.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <omp.h>

ByteTracker::ByteTracker(ByteTrackerConfig config) : _config(config)
{
    switch (config.inputFormat)
    {
    case BoxFormat::TLWH:
    case BoxFormat::CXCYWH:
    case BoxFormat::XYXY:
        break;
    default:
        throw std::invalid_argument("Invalid input box format");
    }
    for (double value : {config.lowConfidence, config.highConfidence, config.newTrackConfidence,
                         config.firstMatchCost, config.secondMatchCost, config.tentativeMatchCost})
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::invalid_argument("ByteTracker thresholds must be in [0, 1]");
    if (config.lowConfidence >= config.highConfidence ||
        config.newTrackConfidence < config.highConfidence)
        throw std::invalid_argument("Invalid ByteTracker confidence ordering or lost buffer");
    if (config.maxLostFrames < 0)
        throw std::invalid_argument("maxLostFrames must be nonnegative");
    if (config.threads < 0)
        throw std::invalid_argument("threads must be nonnegative (0 means auto-detect)");
}

void ByteTracker::reset()
{
    _tracks.clear();
    _nextId = 1;
    _frame = 0;
}

std::vector<std::size_t> ByteTracker::associate(const std::vector<std::size_t> &tracks,
                                                const std::vector<std::size_t> &indices, const std::vector<Detection> &detections,
                                                double maxCost, bool fuseConfidence)
{
    if (tracks.empty() || indices.empty())
        return indices;
    std::vector<Detection> trackBoxes, detectionBoxes;
    trackBoxes.reserve(tracks.size());
    detectionBoxes.reserve(indices.size());
    for (auto i : tracks)
        trackBoxes.push_back(_tracks[i].box());
    for (auto i : indices)
        detectionBoxes.push_back(detections[i]);
    const CostMatrix costs = buildIoUCostMatrix(trackBoxes, detectionBoxes, fuseConfidence);
    const auto matches = _association.solve(costs, maxCost);
    std::vector<bool> used(indices.size(), false);
    for (std::size_t i = 0; i < matches.size(); ++i)
        if (matches[i] >= 0)
        {
            const auto j = static_cast<std::size_t>(matches[i]);
            _tracks[tracks[i]].update(detections[indices[j]]);
            used[j] = true;
        }
    std::vector<std::size_t> remaining;
    for (std::size_t j = 0; j < indices.size(); ++j)
        if (!used[j])
            remaining.push_back(indices[j]);
    return remaining;
}

std::vector<Prediction> ByteTracker::update(const std::vector<Detection> &detections)
{
    ++_frame;
    std::vector<std::size_t> high, low, pool, tentative;
    for (std::size_t i = 0; i < detections.size(); ++i)
        if (detections[i].confidence() >= _config.highConfidence)
            high.push_back(i);
        else if (detections[i].confidence() >= _config.lowConfidence)
            low.push_back(i);

    const int threadLimit = _config.threads > 0 ? _config.threads : omp_get_max_threads();
    const int threadCount = static_cast<int>(std::clamp<std::size_t>(_tracks.size(), 1, threadLimit));

#pragma omp parallel for num_threads(threadCount) if (threadCount > 1)
    for (std::size_t i = 0; i < _tracks.size(); ++i)
    {
        auto &track = _tracks[i];
        // Expire before association so an old ID cannot be resurrected.
        if (track.state() == TrackState::Lost && track.missedFrames() > _config.maxLostFrames)
        {
            track.markRemoved();
            continue;
        }
        track.predict();
    }
    // Build association lists in order after all independent predictions finish.
    for (std::size_t i = 0; i < _tracks.size(); ++i)
    {
        const auto &track = _tracks[i];
        if (track.state() == TrackState::Removed)
            continue;
        if (track.state() == TrackState::Tentative)
            tentative.push_back(i);
        else
            pool.push_back(i);
    }
    high = associate(pool, high, detections, _config.firstMatchCost, true);
    std::vector<std::size_t> remainingActive;
    for (auto i : pool)
        if (_tracks[i].state() == TrackState::Tracked && _tracks[i].missedFrames() > 0)
            remainingActive.push_back(i);
    associate(remainingActive, low, detections, _config.secondMatchCost, false);
    for (auto i : pool)
        if (_tracks[i].missedFrames() > 0)
            _tracks[i].markLost();
    high = associate(tentative, high, detections, _config.tentativeMatchCost, true);
    for (auto i : tentative)
        if (_tracks[i].missedFrames() > 0)
            _tracks[i].markRemoved();
    for (auto i : high)
        if (detections[i].confidence() >= _config.newTrackConfidence)
            _tracks.emplace_back(_nextId++, detections[i], _frame == 1);
    std::erase_if(_tracks, [this](const Track &track)
                  { return track.state() == TrackState::Removed ||
                           (track.state() == TrackState::Lost && track.missedFrames() > _config.maxLostFrames); });
    std::vector<Prediction> predictions;
    for (const auto &track : _tracks)
        if (track.state() == TrackState::Tracked)
            predictions.push_back(track.prediction());
    return predictions;
}
