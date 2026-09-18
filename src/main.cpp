#include "ByteTracker.hpp"
#include <iostream>
#include <memory>

int main()
{
    std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>();
    const std::vector<std::vector<Detection>> frames = {
        {{0, 0, 20, 20, 0.9}},
        {{2, 0, 20, 20, 0.3}}, // Low confidence keeps the existing track.
        {},                    // Temporarily lost.
        {{6, 0, 20, 20, 0.9}}  // Recovered with the same ID.
    };
    for (std::size_t frame = 0; frame < frames.size(); ++frame)
    {
        const auto predictions = tracker->track(frames[frame]);
        std::cout << "Frame " << frame << ": " << predictions.size() << " tracks\n";
        for (const auto &prediction : predictions)
            std::cout << "  ID=" << prediction.id()
                      << " x=" << prediction.box().x()
                      << " y=" << prediction.box().y() << '\n';
    }
}
