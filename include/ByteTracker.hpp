#pragma once

#include "HungarianAlgorithm.hpp"
#include "Track.hpp"
#include "Tracker.hpp"
#include <cstdint>

class ByteTrackerConfig
{
public:
    ByteTrackerConfig(double lowConfidence = 0.1, double highConfidence = 0.6, double newTrackConfidence = 0.7,
                      double firstMatchCost = 0.8, double secondMatchCost = 0.5, double tentativeMatchCost = 0.7,
                      int maxLostFrames = 30, BoxFormat inputFormat = BoxFormat::TLWH, int threads = 0)
        : lowConfidence(lowConfidence), highConfidence(highConfidence), newTrackConfidence(newTrackConfidence),
          firstMatchCost(firstMatchCost), secondMatchCost(secondMatchCost), tentativeMatchCost(tentativeMatchCost),
          maxLostFrames(maxLostFrames), inputFormat(inputFormat), threads(threads) {};

    double lowConfidence;
    double highConfidence;
    double newTrackConfidence;
    double firstMatchCost;
    double secondMatchCost;
    double tentativeMatchCost;
    int maxLostFrames;
    BoxFormat inputFormat;
    int threads = 0; // 0 means auto-detect
};

class ByteTracker final : public Tracker
{
public:
    explicit ByteTracker(ByteTrackerConfig config = {});
    BoxFormat inputFormat() const override { return _config.inputFormat; }
    std::vector<Prediction> update(const std::vector<Detection> &detections) override;
    void reset() override;

private:
    // Updates matched tracks and returns remaining detection indices.
    std::vector<std::size_t> associate(const std::vector<std::size_t> &tracks,
                                       const std::vector<std::size_t> &indices, const std::vector<Detection> &detections,
                                       double maxCost, bool fuseConfidence);
    ByteTrackerConfig _config;
    HungarianAlgorithm _association;
    std::vector<Track> _tracks;
    std::uint64_t _nextId = 1;
    std::uint64_t _frame = 0;
};
