#include "imu_processing.hpp"
#include "utils.h"
using namespace faster_lio;
ImuProcess::ImuProcess() { Q_.setIdentity(); }

ImuProcess::~ImuProcess() {}

void ImuProcess::SetExtrinsic(const Vec3 &transl, const Mat3 &rot) {
    Lidar_T_wrt_IMU_ = transl;
    Lidar_R_wrt_IMU_ = rot;
}

void ImuProcess::AccuImu(const MeasureGroup &meas) {
    /** 1. initializing the gravity_, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity_ **/
    for (int i = 0; i < meas.imu_.size(); i++) {
        const Vec3 &acc = meas.imu_[i].linear_acceleration;
        const Vec3 &omega = meas.imu_[i].angular_velocity;
        if (imu_accu_count == 0) {
            mean_acc_ = acc;
            mean_gyr_ = omega;
            imu_accu_count++;
            continue;
        }
        mean_acc_ += (acc - mean_acc_) / (imu_accu_count + 1);
        mean_gyr_ += (omega - mean_gyr_) / (imu_accu_count + 1);
        imu_accu_count++;
    }
}


void ImuProcess::Predict(const MeasureGroup &meas, StatePoint &state) {
    /*** add the imu_ of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu_;
    v_imu.push_front(last_imu_);
    const double &imu_end_time = v_imu.back().timestamp;
    const double &pcl_end_time = meas.end_time_;

    /*** Initialize IMU pose ***/
    StatePoint imu_state = state;
    imu_poses_.clear();
    imu_poses_.emplace_back(last_lidar_end_time_, imu_state.pos, imu_state.vel_end, acc_last,
                            imu_state.rot.toRotationMatrix(), omega_last);

    /*** forward propagation at each imu_ point ***/
    Vec3 omega_mid, acc_mid;
    double dt = 0;

    ImuInput in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail.timestamp < last_lidar_end_time_) {
            continue;
        }

        omega_mid = 0.5 * (head.angular_velocity + tail.angular_velocity);
        acc_mid = 0.5 * (head.linear_acceleration + tail.linear_acceleration);
        // acc_avr = acc_avr * G_m_s2 / mean_acc_.norm();
        if (head.timestamp < last_lidar_end_time_) {
            dt = tail.timestamp - last_lidar_end_time_;
        } else {
            dt = tail.timestamp - head.timestamp;
        }

        in.acc = acc_mid;
        in.gyro = omega_mid;
        Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
        Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
        Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
        Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
        IESKF::Predict(dt, Q_, in, *state_point_);

        /* save the poses at each IMU measurements */
        imu_state = *state_point_;
        omega_last = omega_mid - imu_state.bias_g;
        acc_last = imu_state.rot * (acc_mid - imu_state.bias_a);
        acc_last += imu_state.gravity;


        imu_poses_.emplace_back(tail.timestamp, imu_state.pos, imu_state.vel_end, acc_last,
                                imu_state.rot.toRotationMatrix(), omega_last);
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    IESKF::Predict(dt, Q_, in, *state_point_);
    last_imu_ = meas.imu_.back();
    last_lidar_end_time_ = pcl_end_time;
}
void ImuProcess::UndistortPoints(StatePoint &state_point, PointCloud::Ptr distort_points, PointCloud &undistort_points) {
    undistort_points = *distort_points;
    std::sort(undistort_points.points.begin(), undistort_points.points.end(),
          [](const faster_lio::Point &x, const faster_lio::Point &y) { return (x.timestamp < y.timestamp); });
    Mat3 R_i;
    Vec3 vel_i, pos_i, acc_i,omega_i;
    double dt;
    /*** undistort each lidar point (backward propagation) ***/
    if (undistort_points.points.empty()) {
        return;
    }
    auto it_k = undistort_points.points.end() - 1;

    Pose3 body_world_end = Pose3(state_point.rot, state_point.pos).GetInverse();
    for (auto imu_i = imu_poses_.end() - 1; imu_i != imu_poses_.begin(); imu_i--) {
        auto head = imu_i - 1;
        auto tail = imu_i;
        R_i = head->rot;
        vel_i = head->vel;
        pos_i = head->pos;
        acc_i = tail->acc;
        omega_i = tail->gyr;

        for (; it_k->timestamp > head->offset_time; it_k--) {
            dt = it_k->timestamp - head->offset_time;

            Mat3 R_k(R_i * ExpMat(omega_i * dt));
            Vec3 pk(it_k->x, it_k->y, it_k->z);
            Vec3 pos_k = pos_i + vel_i * dt + 0.5 * acc_i * dt * dt;
            Vec3 p_compensate = body_world_end * (R_k * pk + pos_k);

            // save Undistorted points and their rotation
            it_k->x = p_compensate(0);
            it_k->y = p_compensate(1);
            it_k->z = p_compensate(2);

            if (it_k == undistort_points.points.begin()) {
                break;
            }
        }
    }
}
void ImuProcess::InertialInitialize(const MeasureGroup &meas, StatePoint &state_point) {
    CHECK(!meas.imu_.empty());
    CHECK(meas.lidar_ != nullptr);
    CHECK(!inertial_initialized);

    /// The very first lidar frame
    AccuImu(meas);
    last_imu_ = meas.imu_.back();
    if (imu_accu_count > MAX_INI_COUNT) {
        // init_state.gravity = S2(-mean_acc_.normalized() * GRAVITY_NORM);
        state_point.gravity = -mean_acc_ / mean_acc_.norm() * GRAVITY_NORM;
        state_point.bias_g = mean_gyr_;

        StatePoint::MatrixN init_P = StatePoint::MatrixN::Identity();
        init_P.block<3, 3>(StatePoint::ROT, StatePoint::ROT) = Mat3::Identity() * 1e-3;
        init_P.block<3, 3>(StatePoint::POS, StatePoint::POS) = Mat3::Identity() * 1e-6;
        init_P.block<3, 3>(StatePoint::VEL, StatePoint::VEL) = Mat3::Identity() * 1e-5;
        init_P.block<3, 3>(StatePoint::BIAS_G, StatePoint::BIAS_G) = Mat3::Identity() * 1e-5;
        init_P.block<3, 3>(StatePoint::BIAS_A, StatePoint::BIAS_A) = Mat3::Identity() * 1e-5;
        init_P.block<2, 2>(StatePoint::GRAVITY, StatePoint::GRAVITY) = Mat2::Identity() * 1e-4;
        state_point.cov = init_P;
        inertial_initialized = true;
        LOG(INFO) << "IMU Initial Done";
    }
}