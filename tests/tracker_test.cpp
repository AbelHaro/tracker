#include "ByteTracker.hpp"
#include <chrono>
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
    require(solver.solve(CostMatrix(0, 0), 0.5).empty(), "Empty assignment");
    require(solver.solve(CostMatrix(2, 0), 0.5) == std::vector<int>({-1, -1}), "No columns");
    require(solver.solve((CostMatrix(2, 2) << 0.1, 0.2, 0.2, 0.9).finished(), 0.5) == std::vector<int>({1, 0}),
            "Assignment must be global and enforce the gate during optimization");
    // Compare small rectangular problems against exhaustive enumeration.
    std::mt19937 random(42);
    for (int n = 1; n <= 4; ++n)
        for (int m = 1; m <= 4; ++m)
            for (int trial = 0; trial < 30; ++trial)
            {
                CostMatrix costs(n, m);
                for (int row = 0; row < n; ++row)
                    for (int col = 0; col < m; ++col)
                        costs(row, col) = (random() % 11) / 10.0;
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
                        if (!(used & (1u << col)) && costs(row, col) <= 0.5)
                            search(row + 1, used | (1u << col), count + 1, total + costs(row, col));
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
                        require(costs(row, col) <= 0.5, "Forbidden assignment");
                        used |= 1u << col;
                        ++count;
                        total += costs(row, col);
                    }
                require(count == bestCount && std::abs(total - bestCost) < 1e-9,
                        "Assignment differs from exhaustive optimum");
            }
}

void testCosts()
{
    std::mt19937 random(17);
    std::vector<Detection> tracks, detections;
    for (int i = 0; i < 37; ++i)
    {
        auto makeBox = [&]()
        {
            const double x = static_cast<int>(random() % 100) - 50;
            const double y = static_cast<int>(random() % 100) - 50;
            const double w = 1 + random() % 60;
            const double h = 1 + random() % 60;
            return Detection(x, y, w, h, (random() % 101) / 100.0);
        };
        tracks.push_back(makeBox());
        detections.push_back(makeBox());
    }
    detections.push_back(tracks.front());
    for (bool fuse : {false, true})
    {
        const auto costs = buildIoUCostMatrix(tracks, detections, fuse);
        for (std::size_t i = 0; i < tracks.size(); ++i)
            for (std::size_t j = 0; j < detections.size(); ++j)
            {
                const double expected = 1 - tracks[i].iou(detections[j]) *
                                                (fuse ? detections[j].confidence() : 1.0);
                require(std::abs(costs(i, j) - expected) < 1e-12,
                        "Eigen costs differ from scalar IoU reference");
            }
    }
    const auto emptyRows = buildIoUCostMatrix({}, detections, false);
    const auto emptyCols = buildIoUCostMatrix(tracks, {}, true);
    require(emptyRows.rows() == 0 && emptyRows.cols() == 38, "Empty row shape");
    require(emptyCols.rows() == 37 && emptyCols.cols() == 0, "Empty column shape");
}

Detection detection(double x, double confidence = 0.9)
{
    return Detection(x, 0, 20, 20, confidence);
}

void testTracking()
{
    Track classified(1, Detection(0, 0, 20, 20, 0.9, 7), true);
    classified.predict();
    require(classified.box().classId() == 7, "Prediction must retain class ID");
    classified.update(Detection(1, 0, 20, 20, 0.8, 3));
    require(classified.box().classId() == 3, "Prediction must use latest detection class ID");

    ByteTrackerConfig config;
    config.maxLostFrames = 2;
    std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>(config);
    auto result = tracker->update({detection(0), detection(100)});
    require(result.size() == 2 && result[0].id() != result[1].id(), "Distinct track IDs");
    const auto first = result[0].id(), second = result[1].id();
    result = tracker->update({detection(101), detection(1, 0.3)});
    require(result.size() == 2 && result[0].id() == first && result[1].id() == second,
            "Reordering and low confidence must preserve IDs");
    require(tracker->update({}).empty(), "Lost tracks must not be emitted");
    result = tracker->update({detection(3), detection(103)});
    require(result.size() == 2 && result[0].id() == first && result[1].id() == second,
            "High confidence must recover lost tracks");
    tracker->update({});
    require(tracker->update({detection(4, 0.3)}).empty(), "Low confidence cannot revive lost track");
    tracker->update({});
    require(tracker->update({detection(4)}).empty(), "Later births require confirmation");
    result = tracker->update({detection(4)});
    require(result.size() == 1 && result[0].id() != first, "Expired ID must not return");
    tracker->reset();
    require(tracker->update({detection(0, 0.3)}).empty(), "Low confidence cannot create tracks");
    require(tracker->update({detection(0)}).empty(), "New track is tentative");
    tracker->update({});
    require(tracker->update({detection(0)}).empty(), "Unconfirmed track must be removed on a miss");
    result = tracker->update({detection(0)});
    require(result.size() == 1, "Repeated high confidence confirms new track");
    tracker->reset();
    result = tracker->update({detection(0)});
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

void testInputFormat()
{
    for (auto format : {BoxFormat::TLWH, BoxFormat::CXCYWH, BoxFormat::XYXY})
    {
        ByteTrackerConfig config;
        config.inputFormat = format;
        std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>(config);
        require(tracker->inputFormat() == format, "Input format must be available through Tracker");
        // Typed C++ detections are already canonical, regardless of the raw input format.
        const auto result = tracker->update({Detection(10, 20, 40, 60, 0.9)});
        require(result[0].box().x() == 10 && result[0].box().y() == 20 &&
                    result[0].box().width() == 40 && result[0].box().height() == 60,
                "Typed Detection must not be converted again");
    }
    ByteTrackerConfig config;
    config.inputFormat = static_cast<BoxFormat>(-1);
    bool rejected = false;
    try
    {
        ByteTracker invalid(config);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected, "Invalid box format must be rejected at construction");
}

void testThreadsPerformance()
{
    constexpr int threadCounts[] = {0, 1, 2, 4, 8};
    constexpr int nDetections = 1000;
    constexpr int nFrames = 100;
    std::vector<Detection> detections;
    detections.reserve(nDetections);
    for (int i = 0; i < nDetections; ++i)
        detections.emplace_back(i * 10, i * 10, 20, 20, 0.9);

    for (int threads : threadCounts)
    {
        ByteTrackerConfig config;
        config.threads = threads; // 0 means auto-detect.
        ByteTracker tracker(config);
        tracker.update(detections); // Create tracks before timing prediction and matching.
        tracker.update(detections); // Warm up the worker threads and association path.

        std::size_t totalTracks = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int frame = 0; frame < nFrames; ++frame)
            totalTracks += tracker.update(detections).size();
        const double elapsedMs = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
        require(totalTracks == nFrames * nDetections, "Thread count changed tracking results");
        std::cout << "threads=" << threads << (threads == 0 ? " (auto)" : "")
                  << " total_ms=" << elapsedMs
                  << " ms/frame=" << elapsedMs / nFrames << '\n';
    }
}

int main()
{
    try
    {
        testCosts();
        testAssignment();
        testTracking();
        testInputFormat();
        testThreadsPerformance();
        std::cout << "All tracker and Hungarian tests passed\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
