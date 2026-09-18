#include "HungarianAlgorithm.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

std::vector<int> HungarianAlgorithm::solve(
    const std::vector<std::vector<double>> &costs, double maxCost) const
{
    if (!std::isfinite(maxCost) || maxCost < 0 || maxCost > 1)
        throw std::invalid_argument("maxCost must be in [0, 1]");
    if (costs.empty())
        return {};
    const int n = static_cast<int>(costs.size());
    const int columns = static_cast<int>(costs.front().size());
    for (const auto &row : costs)
    {
        if (static_cast<int>(row.size()) != columns)
            throw std::invalid_argument("Cost matrix must be rectangular");
        for (double cost : row)
            if (!std::isfinite(cost) || cost < 0)
                throw std::invalid_argument("Costs must be finite and nonnegative");
    }
    // One dummy column per row permits unmatched tracks. The penalty makes
    // cardinality the first priority and total valid cost the second priority.
    const int m = columns + n;
    const double unmatched = n + 1.0;
    const double forbidden = unmatched * (n + 1.0);
    std::vector<double> u(n + 1), v(m + 1);
    std::vector<int> p(m + 1), way(m + 1);
    for (int i = 1; i <= n; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(m + 1, false);
        do
        {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;
            for (int j = 1; j <= m; ++j)
            {
                if (used[j])
                    continue;
                double cost = unmatched;
                if (j <= columns)
                    cost = costs[i0 - 1][j - 1] <= maxCost ? costs[i0 - 1][j - 1] : forbidden;
                const double cur = cost - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j)
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                    minv[j] -= delta;
            j0 = j1;
        } while (p[j0] != 0);
        do
        {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<int> result(n, -1);
    for (int j = 1; j <= columns; ++j)
        if (p[j] && costs[p[j] - 1][j - 1] <= maxCost)
            result[p[j] - 1] = j - 1;
    return result;
}
