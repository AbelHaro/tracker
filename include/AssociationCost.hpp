#pragma once

#include "Detection.hpp"
#include <Eigen/Core>
#include <vector>

// Row-major storage matches the Hungarian solver's row scans.
using CostMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

CostMatrix buildIoUCostMatrix(const std::vector<Detection> &tracks,
                             const std::vector<Detection> &detections,
                             bool fuseConfidence);
