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

    ImuProcess();
    ~ImuProcess();

    void Reset();
    void SetExtrinsic(const Vec3 &transl, const Mat3 &rot);
    void SetGyrCov(const Vec3 &scaler);
    void SetAccCov(const Vec3 &scaler);
    void SetGyrBiasCov(const Vec3 &b_g);
    void SetAccBiasCov(const Vec3 &b_a);
    void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                 PointCloud::Ptr pcl_un_);

    Eigen::Matrix<double, 12, 12> Q_;
    Vec3 cov_acc_;
    Vec3 cov_gyr_;
    Vec3 cov_acc_scale_;
    Vec3 cov_gyr_scale_;
    Vec3 cov_bias_gyr_;
    Vec3 cov_bias_acc_;

   private:
    void IMUInit(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
    void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                      PointCloud &pcl_out);

    PointCloud::Ptr cur_pcl_un_;
    Imu last_imu_;
    std::deque<sensor_msgs::ImuConstPtr> v_imu_;
    std::vector<Pose6D> IMUpose_;
    std::vector<Mat3> v_rot_pcl_;
    Mat3 Lidar_R_wrt_IMU_;
    Vec3 Lidar_T_wrt_IMU_;
    Vec3 mean_acc_;
    Vec3 mean_gyr_;
    Vec3 angvel_last_;
    Vec3 acc_s_last_;
    double last_lidar_end_time_ = 0;
    int init_iter_num_ = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
};

}  // namespace faster_lio

#endif
