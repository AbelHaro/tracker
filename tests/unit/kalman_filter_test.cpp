#include "KalmanFilter.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const char *message)
{
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

void checkCovariance(const KalmanFilter2D &filter)
{
    const auto &p = filter.covariance();
    require(p.allFinite(), "Covariance must be finite");
    require(p.isApprox(p.transpose(), 1e-10), "Covariance must be symmetric");
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigenvalues(p);
    require(eigenvalues.info() == Eigen::Success, "Covariance eigensolver failed");
    require(eigenvalues.eigenvalues().minCoeff() >= -1e-10,
            "Covariance must be positive semidefinite");
}

int main()
{
    try
    {
        KalmanFilter2D initial(1.0, 0.0, 2.0);
        initial.predict();
        require(initial.state().isZero(), "Prediction before initialization must do nothing");
        initial.update(3.0, -4.0);
        require(initial.state().isApprox(Eigen::Vector4d(3, -4, 0, 0)),
                "First measurement must initialize position with zero velocity");
        initial.predict();
        near(initial.x(), 3.0, 1e-12, "Zero velocity must preserve x");
        near(initial.y(), -4.0, 1e-12, "Zero velocity must preserve y");

        // Independent one-step reference: P0 = 10 I, dt = 1, Q = 0, R = 4 I.
        // Predicted position variance = 20; innovation variance = 24.
        // Position gain = 20/24; velocity gain = 10/24.
        KalmanFilter2D reference(1.0, 0.0, 2.0);
        reference.initialize(0, 0);
        reference.predict();
        reference.update(6, -12);
        require(reference.state().isApprox(Eigen::Vector4d(5, -10, 2.5, -5), 1e-12),
                "One-step state differs from hand-calculated result");
        near(reference.covariance()(0, 0), 10.0 / 3.0, 1e-12, "Wrong position variance");
        near(reference.covariance()(0, 2), 5.0 / 3.0, 1e-12, "Wrong cross covariance");
        near(reference.covariance()(2, 2), 35.0 / 6.0, 1e-12, "Wrong velocity variance");
        checkCovariance(reference);

        // Non-unit dt catches accidental assumptions about frame rate.
        constexpr double dt = 0.2;
        KalmanFilter2D moving(dt, 0.1, 0.5);
        moving.initialize(10, -7);
        for (int step = 1; step <= 200; ++step)
        {
            moving.predict();
            moving.update(10 + 3 * step * dt, -7 - 2 * step * dt);
            require(moving.state().allFinite(), "State must remain finite");
            checkCovariance(moving);
        }
        near(moving.x(), 130, 0.01, "Failed to track x");
        near(moving.y(), -87, 0.01, "Failed to track y");
        near(moving.vx(), 3, 0.01, "Failed to learn vx");
        near(moving.vy(), -2, 0.01, "Failed to learn vy");

        const Eigen::Vector4d before = moving.state();
        const double varianceBefore = moving.covariance()(0, 0);
        for (int step = 0; step < 5; ++step)
            moving.predict();
        near(moving.x(), before(0) + 5 * dt * before(2), 1e-10, "Wrong prediction across missing measurements");
        near(moving.y(), before(1) + 5 * dt * before(3), 1e-10, "Wrong y prediction across missing measurements");
        require(moving.covariance()(0, 0) > varianceBefore,
                "Position uncertainty should grow across missing measurements");
        checkCovariance(moving);

        std::cout << "All KalmanFilter2D tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
