#include "Sort.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

Detection box(double x, double confidence = 0.01)
{
    return Detection(x, 0, 20, 20, confidence, 7);
}

int main()
{
    try
    {
        Sort tracker(0.7, 2);
        auto result = tracker.update({box(0), box(100)});
        require(result.size() == 2, "SORT must accept low confidence births");
        const auto first = result[0].id(), second = result[1].id();
        result = tracker.update({box(101), box(1)});
        require(result.size() == 2 && result[0].id() == first && result[1].id() == second,
                "Association must preserve IDs after reordering");
        require(result[0].box().classId() == 7, "Class must be preserved");
        require(tracker.update({}).empty(), "Lost tracks must not be emitted");
        require(tracker.update({}).empty(), "Empty frames must advance age");
        result = tracker.update({box(2)});
        require(result.size() == 1 && result[0].id() == first, "Recover at lost buffer boundary");
        tracker.update({});
        tracker.update({});
        tracker.update({});
        require(tracker.update({box(2)}).empty(), "Later births require confirmation");
        result = tracker.update({box(2)});
        require(result.size() == 1 && result[0].id() != first, "Expired IDs cannot return");
        tracker.reset();
        result = tracker.update({box(0)});
        require(result.size() == 1 && result[0].id() == 1, "Reset restarts IDs and confirmation");
        tracker.reset();
        tracker.update({});
        require(tracker.update({box(0)}).empty(), "Later birth is tentative");
        tracker.update({});
        require(tracker.update({box(0)}).empty(), "Missed tentative must be removed");
        require(tracker.update({box(0)}).size() == 1, "Repeated detection confirms birth");

        Sort strict(0, 0);
        strict.update({box(0)});
        result = strict.update({box(0, 0)});
        require(result.size() == 1 && result[0].id() == 1, "Exact match ignores confidence");
        require(strict.update({box(10)}).empty(), "IoU gate rejects partial overlap");
        result = strict.update({box(10)});
        require(result.size() == 1 && result[0].id() == 2, "Unmatched detection creates new ID");
        strict.update({});
        require(strict.update({box(10)}).empty(), "Zero buffer expires on first miss");

        for (auto format : {BoxFormat::TLWH, BoxFormat::CXCYWH, BoxFormat::XYXY})
        {
            Sort formatted(0.7, 30, format);
            Tracker &base = formatted;
            require(base.inputFormat() == format, "Polymorphic input format");
            require(base.update({box(10)})[0].box().x() == 10, "C++ boxes stay TLWH");
        }
        for (auto [cost, lost, format] : {
                 std::tuple{-0.1, 30, BoxFormat::TLWH},
                 std::tuple{1.1, 30, BoxFormat::TLWH},
                 std::tuple{std::numeric_limits<double>::quiet_NaN(), 30, BoxFormat::TLWH},
                 std::tuple{std::numeric_limits<double>::infinity(), 30, BoxFormat::TLWH},
                 std::tuple{0.7, -1, BoxFormat::TLWH},
                 std::tuple{0.7, 1, static_cast<BoxFormat>(-1)}})
        {
            bool rejected = false;
            try { Sort invalid(cost, lost, format); }
            catch (const std::invalid_argument &) { rejected = true; }
            require(rejected, "Invalid constructor arguments must be rejected");
        }
        std::cout << "All SORT tests passed\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
