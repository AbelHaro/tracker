#include "ByteTracker.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void benchmarkThreadsPerformance()
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
        ByteTracker tracker(0.1, 0.6, 0.7, 0.8, 0.5, 0.7, 30, BoxFormat::TLWH, threads);
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
        benchmarkThreadsPerformance();
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
