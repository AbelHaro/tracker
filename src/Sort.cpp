#include "Sort.hpp"
#include <AssociationCost.hpp>
#include <HungarianAlgorithm.hpp>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>

Sort::Sort(double maxMatchCost, int maxLostFrames, BoxFormat inputFormat)
    : Tracker(inputFormat), _maxMatchCost(maxMatchCost), _maxLostFrames(maxLostFrames)
{
    if (!std::isfinite(maxMatchCost) || maxMatchCost < 0 || maxMatchCost > 1)
        throw std::invalid_argument("maxMatchCost must be in [0, 1]");
    if (maxLostFrames < 0)
        throw std::invalid_argument("maxLostFrames must be nonnegative");
}

void Sort::reset()
{
    _tracks.clear();
    _nextId = 1;
    _frame = 0;
}

std::vector<std::size_t> Sort::associate(const std::vector<std::size_t> &tracks,
                                                const std::vector<std::size_t> &indices, const std::vector<Detection> &detections,
                                                double maxCost)
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
    const CostMatrix costs = buildIoUCostMatrix(trackBoxes, detectionBoxes, false);
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

std::vector<Prediction> Sort::update(const std::vector<Detection> &detections)
{
    ++_frame;
    std::vector<std::size_t> pool;
    for (std::size_t i = 0; i < _tracks.size(); ++i)
    {
        auto &track = _tracks[i];
        // Expire before association so an old ID cannot be resurrected.
        if (track.state() == TrackState::Lost && track.missedFrames() > _maxLostFrames)
        {
            track.markRemoved();
            continue;
        }
        track.predict();
        pool.push_back(i);
    }
    std::vector<std::size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    const auto remaining = associate(pool, indices, detections, _maxMatchCost);
    for (auto i : pool)
        if (_tracks[i].missedFrames() > 0)
        {
            if (_tracks[i].state() == TrackState::Tentative)
                _tracks[i].markRemoved();
            else
                _tracks[i].markLost();
        }
    for (auto i : remaining)
        _tracks.emplace_back(_nextId++, detections[i], _frame == 1);
    std::erase_if(_tracks, [this](const Track &track)
                  { return track.state() == TrackState::Removed ||
                           (track.state() == TrackState::Lost && track.missedFrames() > _maxLostFrames); });
    std::vector<Prediction> predictions;
    for (const auto &track : _tracks)
        if (track.state() == TrackState::Tracked)
            predictions.push_back(track.prediction());
    return predictions;
}
