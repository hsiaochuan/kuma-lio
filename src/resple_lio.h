// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <glog/logging.h>
#include <array>
#include <pcl/common/transforms.h>

#include "common_lib.h"
#include "eigen_type.h"
#include "ikd-Tree/ikd_Tree.h"
#include "resple_param.h"
#include "spline_state.h"
#include "timer.h"
namespace kuma_lio {
// window control points (4 x 6) | ba | bg
constexpr int XSIZE = 30;
struct PointData {
    Vec3 point_body;
    Vec3 point_world;
    kuma_lio::Point point;
    int64_t time_ns;
    Vec3 ppl_normal;
    double ppl_d;
    double z;
    PointVector near_points;  // the map neighbours of the last match, reused when the scan goes into the map
    Eigen::Matrix<double, 1, 24> H;
    bool if_valid = false;
    double pt_wt = 0.01;
};
struct ImuData {
    int64_t time_ns;
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
    Eigen::Matrix<double, 6, XSIZE> H;
    Eigen::Matrix<double, 6, 1> imu_itp;
    Eigen::Matrix<double, 6, 1> z;
    // per-axis outlier gating, decided in prepIMU
    std::array<bool, 3> if_acc_valid = {false, false, false};
    std::array<bool, 3> if_gyro_valid = {false, false, false};
    ImuData() {}
    ImuData(const int64_t s, const Eigen::Vector3d& w, const Eigen::Vector3d& a) : time_ns(s), gyro(w), accel(a) {}

    ImuData(const ImuData& other)
        : time_ns(other.time_ns), gyro(other.gyro), accel(other.accel), H(other.H), imu_itp(other.imu_itp),
          z(other.z), if_acc_valid(other.if_acc_valid), if_gyro_valid(other.if_gyro_valid) {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
class RESPLE_LIO {
   public:
    static constexpr int CP_SIZE = 24;
    static constexpr int S1 = 0, S2 = 6, S3 = 12, S4 = 18;
    static constexpr int Q1 = 3, Q2 = 9, Q3 = 15, Q4 = 21;
    static constexpr int BA_OFFSET = 24;
    static constexpr int BG_OFFSET = 27;
    static constexpr double GRAVITY_NORM = 9.81;
    static constexpr int MATCH_POINT = 5;
    // below this many points a loop stays serial; a 10 ms knot holds ~250 points and the ikd-tree search wants threads
    static constexpr int kMinParallelPoints = 128;
    void Init();

    void ProcessMeasurement(MeasureGroup & measures);

    Eigen::Matrix<double, XSIZE, 1> getState();
    static bool esti_plane(Vec4& pca_result, const PointVector& point, const double& threshold);
    int Match();
    void IterativeUpdate();
    void prepIMU(ImuData& imu_data);
    void prepLiDAR(PointData& pt_data) const;
    void propRCP();
    void UpdateLiDARInertial(int eff_lidar_num, const Eigen::Matrix<double, XSIZE, 1>& x_prop,
                             const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop);

    template <int RSIZE>
    void Update(const Eigen::Matrix<double, RSIZE, 1>& innov, const Eigen::Matrix<double, RSIZE, 1>& R_inv,
                const Eigen::Matrix<double, RSIZE, XSIZE>& H, const Eigen::Matrix<double, XSIZE, 1>& x_prop,
                const Eigen::Matrix<double, XSIZE, XSIZE>& cov_prop) {
        int num_pts = innov.rows();
        Eigen::Matrix<double, XSIZE, 1> RCPs_post;
        Eigen::MatrixXd I_X = Eigen::MatrixXd::Identity(XSIZE, XSIZE);
        if (num_pts > XSIZE) {
            Eigen::Matrix<double, XSIZE, XSIZE> cov_rcp_inv = cov_prop.llt().solve(I_X);
            Eigen::Matrix<double, XSIZE, RSIZE> HT_R_inv;
            HT_R_inv.noalias() = (H.transpose().array().rowwise() * R_inv.transpose().array()).matrix();
            Eigen::Matrix<double, XSIZE, XSIZE> HT_R_inv_H;
            HT_R_inv_H.noalias() = HT_R_inv * H;

            Eigen::Matrix<double, XSIZE, XSIZE> S = HT_R_inv_H;
            S.noalias() += cov_rcp_inv;
            Eigen::Matrix<double, XSIZE, XSIZE> S_inv = S.llt().solve(I_X);
            Eigen::Matrix<double, XSIZE, RSIZE> K;
            K.noalias() = S_inv * HT_R_inv;

            KH.noalias() = S_inv * HT_R_inv_H;
            Eigen::Matrix<double, XSIZE, 1> delta_cur = (getState() - x_prop);
            Eigen::Matrix<double, XSIZE, 1> deltax = KH * delta_cur + K * innov - delta_cur;
            RCPs_post.noalias() = getState() + deltax;
        } else {
            Eigen::Matrix<double, RSIZE, RSIZE> R = R_inv.cwiseInverse().asDiagonal();
            Eigen::Matrix<double, RSIZE, RSIZE> S;
            S.noalias() = H * cov_prop * H.transpose() + R;
            Eigen::Matrix<double, XSIZE, RSIZE> K;
            K.noalias() = cov_prop * H.transpose() * S.inverse();
            KH.noalias() = K * H;
            Eigen::Matrix<double, XSIZE, 1> delta_cur = (getState() - x_prop);
            Eigen::Matrix<double, XSIZE, 1> deltax = KH * delta_cur + K * innov - delta_cur;
            RCPs_post.noalias() = getState() + deltax;
        }
        UpdateState(RCPs_post);
    }
    void UpdateState(const Eigen::Matrix<double, XSIZE, 1>& xupd);

    SplineState spl;
    Eigen::Matrix<double, XSIZE, XSIZE> P;
    Eigen::Matrix<double, XSIZE, XSIZE> Q_mat;
    Eigen::Matrix<double, XSIZE, XSIZE> A_mat;
    Eigen::Matrix<double, XSIZE, XSIZE> KH;
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, GRAVITY_NORM);
    std::shared_ptr<RespleParam> param;

    // The ikd-tree local map Match() queries: the registered scans go in every 0.1 s (MapIncremental), and the
    // map cube slides with the sensor (LaserMapFovSegment).
    std::shared_ptr<KD_TREE<Point>> kd_tree;
    void MapIncremental();
    void LaserMapFovSegment();
    BoxPointType localmap_box;
    bool localmap_box_initialized = false;
    PointCloud accum_points;                       // world frame, since the last map update
    std::vector<PointVector> accum_nearest_points;  // their neighbours from the last match

    Eigen::aligned_deque<ImuData> imu_meas;
    Eigen::aligned_deque<PointData> pt_meas;
};
}  // namespace kuma_lio
