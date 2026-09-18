#include "AssociationCost.hpp"

CostMatrix buildIoUCostMatrix(const std::vector<Detection> &tracks,
                              const std::vector<Detection> &detections,
                              bool fuseConfidence)
{
    const Eigen::Index columns = static_cast<Eigen::Index>(detections.size());
    CostMatrix costs(static_cast<Eigen::Index>(tracks.size()), columns);
    if (tracks.empty() || detections.empty())
        return costs;

    // Each row is contiguous, allowing coefficient-wise SIMD operations.
    Eigen::Array<double, 6, Eigen::Dynamic, Eigen::RowMajor> boxes(6, columns);
    for (Eigen::Index j = 0; j < columns; ++j)
    {
        const auto &box = detections[static_cast<std::size_t>(j)];
        boxes.col(j) << box.x(), box.y(), box.x() + box.width(),
            box.y() + box.height(), box.width() * box.height(), box.confidence();
    }
    Eigen::Array<double, 1, Eigen::Dynamic> intersection(columns);
    for (Eigen::Index i = 0; i < costs.rows(); ++i)
    {
        const auto &box = tracks[static_cast<std::size_t>(i)];
        intersection =
            (boxes.row(2).min(box.x() + box.width()) - boxes.row(0).max(box.x())).max(0.0) *
            (boxes.row(3).min(box.y() + box.height()) - boxes.row(1).max(box.y())).max(0.0);
        auto row = costs.row(i).array();
        row = intersection / (boxes.row(4) + box.width() * box.height() - intersection);
        if (fuseConfidence)
            row *= boxes.row(5);
        row = 1.0 - row;
    }
    return costs;
}
