#pragma once

#include <vector>

class HungarianAlgorithm
{
public:
    // Rectangular nonnegative costs. Returns a column per row, or -1.
    // Costs above maxCost are forbidden during optimization.
    std::vector<int> solve(const std::vector<std::vector<double>> &costs,
                           double maxCost) const;
};
