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
    KalmanFilter2D() : KalmanFilter2D(1.0, 2.0, 2.0) {} // Default constructor

    KalmanFilter2D(double dt, double processNoiseStd, double measurementNoiseStd) : _dt(dt)
    {
        _x.setZero();

        // State transition matrix
        _F << 1.0, 0.0, dt, 0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;

        _P << 10.0, 0.0, 0.0, 0.0,
            0.0, 10.0, 0.0, 0.0,
            0.0, 0.0, 10.0, 0.0,
            0.0, 0.0, 0.0, 10.0;

        const double sigmaA2 = processNoiseStd * processNoiseStd;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt3 * dt;

        _Q = sigmaA2 * (Eigen::Matrix4d() << dt4 / 4.0, 0.0, dt3 / 2.0, 0.0,
                        0.0, dt4 / 4.0, 0.0, dt3 / 2.0,
                        dt3 / 2.0, 0.0, dt2, 0.0,
                        0.0, dt3 / 2.0, 0.0, dt2)
                           .finished();

        const double sigmaM2 = measurementNoiseStd * measurementNoiseStd;

        _R << sigmaM2, 0.0,
            0.0, sigmaM2;
    }

    void initialize(double posX, double posY)
    {
        _x << posX,
            posY,
            0.0,
            0.0;
        initialized = true;
    }

    void predict()
    {
        if (!initialized)
            return;

        // x^ = F * x
        _x.head<2>() += _dt * _x.tail<2>();

        // P^ = F * P * F^T + Q
        const Eigen::Matrix4d FP = _F * _P;
        _P.noalias() = FP * _F.transpose();
        _P += _Q;
    }

    void update(double measX, double measY)
    {
        if (!initialized)
        {
            initialize(measX, measY);
            return;
        }

        Eigen::Vector2d z;
        z << measX, measY;

        // Innovation
        // y = z - H * x
        const Eigen::Vector2d innovation = z - _x.head<2>();

        // Innovation covariance
        // S = H * P * H^T + R
        const Eigen::Matrix2d S = _P.topLeftCorner<2, 2>() + _R;

        // Kalman gain
        const Eigen::Matrix<double, 4, 2> PHt = _P.leftCols<2>();
        Eigen::Matrix<double, 4, 2> K = S.ldlt().solve(PHt.transpose()).transpose();

        _x.noalias() += K * innovation;

        // Joseph form update for covariance
        Eigen::Matrix4d IKH = Eigen::Matrix4d::Identity();
        IKH.leftCols<2>() -= K; // H = [I2, 0].
        const Eigen::Matrix4d corrected = IKH * _P;
        const Eigen::Matrix<double, 4, 2> KR = K * _R;
        _P.noalias() = corrected * IKH.transpose();
        _P.noalias() += KR * K.transpose();
    }

    double x() const { return _x(0); }
    double y() const { return _x(1); }
    double vx() const { return _x(2); }
    double vy() const { return _x(3); }

    const Eigen::Vector4d &state() const { return _x; }
    const Eigen::Matrix4d &covariance() const { return _P; }
};
