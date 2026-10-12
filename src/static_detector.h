//
// Whether the IMU is at rest, judged against its own noise at start-up.
//

#ifndef FASTER_LIO_STATIC_DETECTOR_H
#define FASTER_LIO_STATIC_DETECTOR_H
#include <algorithm>
#include <cmath>
#include <deque>

#include "common_lib.h"

namespace faster_lio {

struct ZuptParam {
    bool enable = false;
    double window = 0.5;       ///< [s] of IMU samples and of trajectory that must be still
    double noise_ratio = 2.0;  ///< the window's IMU scatter may reach this many times the scatter at start-up
    double peak_ratio = 4.0;   ///< and a single sample may leave the window mean by this many times it
    double max_move = 0.01;    ///< [m] the estimated pose may move this far over the window
    double max_turn = 0.005;   ///< [rad] and turn this far
    double std_vel = 0.01;     ///< [m/s] noise of the zero-velocity measurement
    double std_gyr = 0.002;    ///< [rad/s] noise of the zero-angular-rate measurement
};

/// The IMU is quiet when a whole window of samples scatters like the samples of the start-up, which
/// the system takes to be at rest anyway (gravity and the gyroscope bias come from it).  The
/// thresholds are therefore the sensor's own: a drone with idling motors and a handheld unit on
/// the floor each get theirs.  Quiet is not yet static - a steady glide is quiet too - so the
/// caller also asks the trajectory.
class StaticDetector {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit StaticDetector(const ZuptParam& param = ZuptParam()) : param_(param) {}

    /// Scatter at rest from the samples of `imus` before `end_ns`.
    void SetReference(const std::deque<Imu>& imus, int64_t end_ns) {
        std::deque<Imu> rest;
        for (const Imu& imu : imus)
            if (imu.time_ns < end_ns) rest.push_back(imu);
        if (rest.size() < 10) return;
        Vec3 acc_mean, gyr_mean;
        Scatter(rest, &acc_mean, &gyr_mean, &acc_ref_, &gyr_ref_);
        // a start-up that was not at rest must not loosen the thresholds; nor may a noiseless one close them
        acc_ref_ = std::min(std::max(acc_ref_, kAccFloor), kAccCap);
        gyr_ref_ = std::min(std::max(gyr_ref_, kGyrFloor), kGyrCap);
        has_reference_ = true;
    }

    void Add(const Imu& imu) {
        window_.push_back(imu);
        while (window_.back().time_ns - window_.front().time_ns > S_TO_NS(param_.window)) window_.pop_front();
    }

    bool Quiet() const {
        if (!has_reference_ || window_.size() < 10) return false;
        if (window_.back().time_ns - window_.front().time_ns < S_TO_NS(0.9 * param_.window)) return false;
        Vec3 acc_mean, gyr_mean;
        double acc_std, gyr_std;
        Scatter(window_, &acc_mean, &gyr_mean, &acc_std, &gyr_std);
        if (acc_std > param_.noise_ratio * acc_ref_ || gyr_std > param_.noise_ratio * gyr_ref_) return false;
        for (const Imu& imu : window_) {
            if ((imu.linear_acceleration - acc_mean).norm() > param_.peak_ratio * acc_ref_) return false;
            if ((imu.angular_velocity - gyr_mean).norm() > param_.peak_ratio * gyr_ref_) return false;
        }
        return true;
    }

    double AccReference() const { return acc_ref_; }
    double GyrReference() const { return gyr_ref_; }

   private:
    static constexpr double kAccFloor = 0.02, kAccCap = 0.5;    ///< [m/s^2]
    static constexpr double kGyrFloor = 0.002, kGyrCap = 0.05;  ///< [rad/s]

    /// Means, and the norms of the per-axis standard deviations.
    static void Scatter(const std::deque<Imu>& imus, Vec3* acc_mean, Vec3* gyr_mean, double* acc_std,
                        double* gyr_std) {
        Vec3 acc_sum = Vec3::Zero(), gyr_sum = Vec3::Zero(), acc_sq = Vec3::Zero(), gyr_sq = Vec3::Zero();
        for (const Imu& imu : imus) {
            acc_sum += imu.linear_acceleration;
            gyr_sum += imu.angular_velocity;
            acc_sq += imu.linear_acceleration.cwiseAbs2();
            gyr_sq += imu.angular_velocity.cwiseAbs2();
        }
        const double n = static_cast<double>(imus.size());
        *acc_mean = acc_sum / n;
        *gyr_mean = gyr_sum / n;
        *acc_std = std::sqrt(std::max(0.0, (acc_sq / n - acc_mean->cwiseAbs2()).sum()));
        *gyr_std = std::sqrt(std::max(0.0, (gyr_sq / n - gyr_mean->cwiseAbs2()).sum()));
    }

    ZuptParam param_;
    std::deque<Imu> window_;
    double acc_ref_ = 0.0, gyr_ref_ = 0.0;
    bool has_reference_ = false;
};

}  // namespace faster_lio
#endif  // FASTER_LIO_STATIC_DETECTOR_H
