#pragma once

#include "AssociationCost.hpp"
#include <vector>

class HungarianAlgorithm
{
public:
    // Rectangular nonnegative costs. Returns a column per row, or -1.
    // Costs above maxCost are forbidden during optimization.
    std::vector<int> solve(const Eigen::Ref<const CostMatrix> &costs,
                           double maxCost) const;
};
