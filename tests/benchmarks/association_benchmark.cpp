#include "AssociationCost.hpp"
#include <chrono>
#include <iostream>
#include <random>

int main()
{
    std::mt19937 random(17);
    double checksum = 0;
    for (int size : {16, 64, 256})
    {
        std::vector<Detection> boxes;
        for (int i = 0; i < size; ++i)
            boxes.emplace_back(random() % 100, random() % 100, 20, 20, 0.9);
        const int repeats = 1000;
        auto start = std::chrono::steady_clock::now();
        for (int k = 0; k < repeats; ++k)
        {
            std::vector<std::vector<double>> costs(size, std::vector<double>(size));
            for (int i = 0; i < size; ++i)
                for (int j = 0; j < size; ++j)
                    costs[i][j] = 1 - boxes[i].iou(boxes[j]) * boxes[j].confidence();
            checksum += costs[k % size][(k + 1) % size];
        }
        auto middle = std::chrono::steady_clock::now();
        for (int k = 0; k < repeats; ++k)
        {
            auto costs = buildIoUCostMatrix(boxes, boxes, true);
            checksum += costs(k % size, (k + 1) % size);
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << size << "x" << size << " scalar_ms="
                  << std::chrono::duration<double, std::milli>(middle - start).count()
                  << " eigen_ms=" << std::chrono::duration<double, std::milli>(end - middle).count() << '\n';
    }
    std::cout << "checksum=" << checksum << '\n';
}
