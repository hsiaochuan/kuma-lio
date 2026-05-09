#ifndef FASTER_LIO_IMU_PROCESSING_H
#define FASTER_LIO_IMU_PROCESSING_H

#include <sensor_msgs/Imu.h>
#include <cmath>
#include <deque>
#include "common_lib.h"
#include "use-ikfom.hpp"

namespace faster_lio {

constexpr int MAX_INI_COUNT = 20;

/// IMU Process and undistortion
class ImuProcess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct PoseWithVel {
        PoseWithVel() : pos(Vec3::Zero()), vel(Vec3::Zero()), acc(Vec3::Zero()), rot(Mat3::Identity()), gyr(Vec3::Zero()) {}
        PoseWithVel(const double & offset, const Vec3 &p, const Vec3 &v, const Vec3 &a, const Mat3 &r, const Vec3 &g)
            : offset_time(offset), pos(p), vel(v), acc(a), rot(r), gyr(g) {}
        double offset_time = 0;
        Vec3 pos;
        Vec3 vel;
        Vec3 acc;

        Mat3 rot;
        Vec3 gyr;
    };
    ImuProcess();
    ~ImuProcess();

    void SetExtrinsic(const Vec3 &transl, const Mat3 &rot);
    void SetGyrCov(const Vec3 &scaler);
    void SetAccCov(const Vec3 &scaler);
    void SetGyrBiasCov(const Vec3 &b_g);
    void SetAccBiasCov(const Vec3 &b_a);
    void InertialInitialize(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state);

    Eigen::Matrix<double, 12, 12> Q_;
    Vec3 cov_acc_;
    Vec3 cov_gyr_;
    Vec3 cov_acc_scale_;
    Vec3 cov_gyr_scale_;
    Vec3 cov_bias_gyr_;
    Vec3 cov_bias_acc_;

    void AccuImu(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state);
    void PredictAndUndistort(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state);
    void UndistortPoints(esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloud::Ptr distort_points, PointCloud& undistort_points);
    Imu last_imu_;
    std::vector<PoseWithVel> imu_poses_;

    Mat3 Lidar_R_wrt_IMU_ = Mat3::Identity();
    Vec3 Lidar_T_wrt_IMU_ = Vec3::Zero();
    Vec3 mean_acc_ = Vec3::Zero();
    Vec3 mean_gyr_ = Vec3::Zero();

    Vec3 omega_last = Vec3::Zero();
    Vec3 acc_last = Vec3::Zero();

    double last_lidar_end_time_ = 0;
    int imu_accu_count = 0;
    bool inertial_initialized = false;
};

}  // namespace faster_lio

#endif
