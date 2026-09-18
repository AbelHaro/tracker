#include <iostream>
#include <Eigen/Dense>
#include "KalmanFilter.hpp"

double lineal(double x)
{
    return x;
}
int main()
{
    std::cout << "Hello World\n";

    double dt = 1;

    double processNoiseStd = 2.0;
    double measurementNoiseStd = 2.0;

    KalmanFilter2D kf(dt, processNoiseStd, measurementNoiseStd);

    kf.initialize(320.0, 240.0);

    Eigen::Vector2d detections[] = {
        {322.0, 238.0},
        {324.0, 236.0},
        {326.0, 234.0},
        {328.0, 232.0},
        {330.0, 230.0}};

    for (const auto &detection : detections)
    {
        kf.predict();

        std::cout << "Predicted state: x = " << kf.x() << ", y = " << kf.y()
                  << ", vx = " << kf.vx() << ", vy = " << kf.vy() << std::endl;

        kf.update(detection.x(), detection.y());

        std::cout << "Updated state: x = " << kf.x() << ", y = " << kf.y()
                  << ", vx = " << kf.vx() << ", vy = " << kf.vy() << std::endl;
    }

    KalmanFilter2D kf2(dt, processNoiseStd, measurementNoiseStd);

    const int NUM_STEPS = 10;
    Eigen::Vector2d measurements[NUM_STEPS] = {
        {1.0, lineal(1.0)},
        {2.0, lineal(2.0)},
        {3.0, lineal(3.0)},
        {4.0, lineal(4.0)},
        {5.0, lineal(5.0)},
        {6.0, lineal(6.0)},
        {7.0, lineal(7.0)},
        {8.0, lineal(8.0)},
        {9.0, lineal(9.0)},
        {10.0, lineal(10.0)}};

    for (const auto &measurement : measurements)
    {
        kf2.predict();

        std::cout << "Predicted state: x = " << kf2.x() << ", y = " << kf2.y()
                  << ", vx = " << kf2.vx() << ", vy = " << kf2.vy() << std::endl;

        kf2.update(measurement.x(), measurement.y());

        std::cout << "Updated state: x = " << kf2.x() << ", y = " << kf2.y()
                  << ", vx = " << kf2.vx() << ", vy = " << kf2.vy() << std::endl;
    }

    return 0;
}