#pragma once
#include "Detection.hpp"
#include "Prediction.hpp"
#include "Track.hpp"
#include "Tracker.hpp"
#include "HungarianAlgorithm.hpp"

class Sort final : public Tracker
{
public:
    // maxMatchCost limits 1 - IoU; 0.7 requires IoU >= 0.3.
    explicit Sort(double maxMatchCost = 0.7, int maxLostFrames = 30, BoxFormat inputFormat = BoxFormat::TLWH);
    std::vector<Prediction> update(const std::vector<Detection> &detections) override;
    void reset() override;

private:
    std::vector<std::size_t> associate(const std::vector<std::size_t> &tracks,
                                       const std::vector<std::size_t> &indices, const std::vector<Detection> &detections,
                                       double maxCost);

    double _maxMatchCost;
    int _maxLostFrames;
    HungarianAlgorithm _association;
    std::uint64_t _nextId = 1;
    std::uint64_t _frame = 0;
    std::vector<Track> _tracks;
};
