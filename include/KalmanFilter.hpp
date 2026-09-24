#pragma once

#include <Eigen/Dense>

class KalmanFilter2D
{
private:
    double _dt;               // Time step
    bool initialized = false; // Flag to check if the filter is initialized
    // [X, Y, Vx, Vy]
    Eigen::Vector4d _x;

    // State transition matrix
    Eigen::Matrix4d _F;

    // State covariance matrix
    Eigen::Matrix4d _P;

    // Process noise covariance matrix
    Eigen::Matrix4d _Q;

    // Measurement noise matrix
    Eigen::Matrix2d _R;

public:
    KalmanFilter2D();
    KalmanFilter2D(double dt, double processNoiseStd, double measurementNoiseStd);
    void initialize(double posX, double posY);
    void predict();
    void update(double measX, double measY);
    double x() const;
    double y() const;
    double vx() const;
    double vy() const;
    const Eigen::Vector4d &state() const;
    const Eigen::Matrix4d &covariance() const;
};
