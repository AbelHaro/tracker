#include "ByteTracker.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void testAssignment()
{
    HungarianAlgorithm solver;
    require(solver.solve({}, 0.5).empty(), "Empty assignment");
    require(solver.solve({{}, {}}, 0.5) == std::vector<int>({-1, -1}), "No columns");
    require(solver.solve({{0.1, 0.2}, {0.2, 0.9}}, 0.5) == std::vector<int>({1, 0}),
            "Assignment must be global and enforce the gate during optimization");
    // Compare small rectangular problems against exhaustive enumeration.
    std::mt19937 random(42);
    for (int n = 1; n <= 4; ++n)
        for (int m = 1; m <= 4; ++m)
            for (int trial = 0; trial < 30; ++trial)
            {
                std::vector<std::vector<double>> costs(n, std::vector<double>(m));
                for (auto &row : costs)
                    for (auto &cost : row)
                        cost = (random() % 11) / 10.0;
                int bestCount = -1;
                double bestCost = 0;
                std::function<void(int, unsigned, int, double)> search =
                    [&](int row, unsigned used, int count, double total)
                {
                    if (row == n)
                    {
                        if (count > bestCount || (count == bestCount && total < bestCost))
                        {
                            bestCount = count;
                            bestCost = total;
                        }
                        return;
                    }
                    search(row + 1, used, count, total);
                    for (int col = 0; col < m; ++col)
                        if (!(used & (1u << col)) && costs[row][col] <= 0.5)
                            search(row + 1, used | (1u << col), count + 1, total + costs[row][col]);
                };
                search(0, 0, 0, 0);
                auto assignment = solver.solve(costs, 0.5);
                unsigned used = 0;
                int count = 0;
                double total = 0;
                for (int row = 0; row < n; ++row)
                    if (assignment[row] >= 0)
                    {
                        const int col = assignment[row];
                        require(!(used & (1u << col)), "Duplicate assigned column");
                        require(costs[row][col] <= 0.5, "Forbidden assignment");
                        used |= 1u << col;
                        ++count;
                        total += costs[row][col];
                    }
                require(count == bestCount && std::abs(total - bestCost) < 1e-9,
                        "Assignment differs from exhaustive optimum");
            }
}

Detection detection(double x, double confidence = 0.9)
{
    return Detection(x, 0, 20, 20, confidence);
}

void testTracking()
{
    ByteTrackerConfig config;
    config.maxLostFrames = 2;
    std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>(config);
    auto result = tracker->track({detection(0), detection(100)});
    require(result.size() == 2 && result[0].id() != result[1].id(), "Distinct track IDs");
    const auto first = result[0].id(), second = result[1].id();
    result = tracker->track({detection(101), detection(1, 0.3)});
    require(result.size() == 2 && result[0].id() == first && result[1].id() == second,
            "Reordering and low confidence must preserve IDs");
    require(tracker->track({}).empty(), "Lost tracks must not be emitted");
    result = tracker->track({detection(3), detection(103)});
    require(result.size() == 2 && result[0].id() == first && result[1].id() == second,
            "High confidence must recover lost tracks");
    tracker->track({});
    require(tracker->track({detection(4, 0.3)}).empty(), "Low confidence cannot revive lost track");
    tracker->track({});
    require(tracker->track({detection(4)}).empty(), "Later births require confirmation");
    result = tracker->track({detection(4)});
    require(result.size() == 1 && result[0].id() != first, "Expired ID must not return");
    tracker->reset();
    require(tracker->track({detection(0, 0.3)}).empty(), "Low confidence cannot create tracks");
    require(tracker->track({detection(0)}).empty(), "New track is tentative");
    tracker->track({});
    require(tracker->track({detection(0)}).empty(), "Unconfirmed track must be removed on a miss");
    result = tracker->track({detection(0)});
    require(result.size() == 1, "Repeated high confidence confirms new track");
    tracker->reset();
    result = tracker->track({detection(0)});
    require(result.size() == 1 && result[0].id() == 1, "Reset starts a new ID sequence");
    require(detection(0).iou(detection(0)) == 1 && detection(0).iou(detection(100)) == 0,
            "Wrong IoU");
    bool rejected = false;
    try
    {
        Detection invalid(0, 0, -1, 2, 0.5);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected, "Invalid detection must be rejected");
    rejected = false;
    config.lowConfidence = config.highConfidence;
    try
    {
        ByteTracker invalid(config);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected, "Invalid config must be rejected");
}

int main()
{
    try
    {
        testAssignment();
        testTracking();
        std::cout << "All tracker and Hungarian tests passed\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
