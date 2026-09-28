#pragma once
#include "Detection.hpp"
#include "Prediction.hpp"
#include "Track.hpp"
#include "Tracker.hpp"
#include "HungarianAlgorithm.hpp"

class SortConfig
{
public:
    SortConfig(double maxIoU = 0.7, int maxLostFrames = 30, BoxFormat inputFormat = BoxFormat::TLWH)
        : maxIoU(maxIoU), maxLostFrames(maxLostFrames), inputFormat(inputFormat) {};

    // Maximum association cost (1 - IoU), without confidence fusion.
    double maxIoU;
    int maxLostFrames;
    BoxFormat inputFormat;
};

class Sort final : public Tracker
{
public:
    explicit Sort(SortConfig config = {});
    std::vector<Prediction> update(const std::vector<Detection> &detections) override;
    void reset() override;

private:
    std::vector<std::size_t> associate(const std::vector<std::size_t> &tracks,
                                       const std::vector<std::size_t> &indices, const std::vector<Detection> &detections,
                                       double maxIoU);

    SortConfig _config;
    HungarianAlgorithm _association;
    std::uint64_t _nextId = 1;
    std::uint64_t _frame = 0;
    std::vector<Track> _tracks;
};
