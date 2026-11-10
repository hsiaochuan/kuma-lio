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
#include <memory>
#include <vector>

#include <deque>
#include <unordered_map>

#include "common_lib.h"
#include "eigen_type.h"
#include "gnss_epoch.h"
#include "laser_mapping_param.h"
#include "local_ba.h"

namespace kuma_lio {

// Voxel-SLAM IMUST: the IMU frame in the world, with a fixed gravity outside the state.
struct FastLioState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int DIM = 15;  // R p v bg ba
    using MatN = Eigen::Matrix<double, DIM, DIM>;
    using VecN = Eigen::Matrix<double, DIM, 1>;

    FastLioState() {
        cov = MatN::Identity() * 1e-4;
        cov.block<6, 6>(9, 9) = Eigen::Matrix<double, 6, 6>::Identity() * 1e-5;
    }
    FastLioState& operator+=(const VecN& dx) {
        R = R * ExpMat(Vec3(dx.segment<3>(0)));
        p += dx.segment<3>(3);
        v += dx.segment<3>(6);
        bg += dx.segment<3>(9);
        ba += dx.segment<3>(12);
        return *this;
    }
    Eigen::Isometry3d Isometry() const {
        Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
        iso.linear() = R;
        iso.translation() = p;
        return iso;
    }
    VecN operator-(const FastLioState& b) const {
        VecN dx;
        const Mat3 dR = b.R.transpose() * R;
        dx.segment<3>(0) = LogMat(dR);
        dx.segment<3>(3) = p - b.p;
        dx.segment<3>(6) = v - b.v;
        dx.segment<3>(9) = bg - b.bg;
        dx.segment<3>(12) = ba - b.ba;
        return dx;
    }

    Mat3 R = Mat3::Identity();
    Vec3 p = Vec3::Zero();
    Vec3 v = Vec3::Zero();
    Vec3 bg = Vec3::Zero();
    Vec3 ba = Vec3::Zero();
    MatN cov;
};

struct RTKState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int DIM = 8;  // clk (4) | yaw | t_ecef_enu (3)
    static constexpr int CLK = 0, YAW = 4, TRN = 5;
    using Vec = Eigen::Matrix<double, DIM, 1>;

    Vec Vector() const {
        Vec v;
        v << clk, yaw, t_ecef_enu;
        return v;
    }
    void SetVector(const Vec& v) {
        clk = v.segment<4>(CLK);
        yaw = v(YAW);
        t_ecef_enu = v.segment<3>(TRN);
    }
    // R_ecef_enu = R_ecef_enu_init * Rz(yaw)
    Mat3 RotEcefEnu() const { return R_ecef_enu_init * Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(); }

    bool aligned = false;
    Vec4 clk = Vec4::Zero();         // c * receiver clock [m]: GPS, GLO, GAL, BDS
    Vec4 clk_anchor = Vec4::Zero();  // the clocks at the anchor epoch, held fixed
    double yaw = 0.0;
    Vec3 t_ecef_enu = Vec3::Zero();
    Mat3 R_ecef_enu_init = Mat3::Identity();
    Eigen::Matrix<double, DIM, DIM> P = Eigen::Matrix<double, DIM, DIM>::Identity();
    Eigen::Matrix<double, FastLioState::DIM, DIM> P_ng = Eigen::Matrix<double, FastLioState::DIM, DIM>::Zero();
};

struct FastLioPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 body = Vec3::Zero();   // undistorted to the end of the frame, IMU frame
    Mat3 var = Mat3::Zero();    // covariance of `body`
    Vec3 world = Vec3::Zero();
    Vec3 normal = Vec3::Zero();
    double residual = 0.0;
    double r_inv = 0.0;
    bool valid = false;         // matched a plane
};

class FastLio {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr double GRAVITY_NORM = 9.81;

    void Init();
    // Start from `state` at `time_ns`; `last_imu` is the last sample before it.
    void Start(const FastLioState& state, const Vec3& gravity_up, int64_t time_ns, const Imu& last_imu);
    // One frame: everything between the last `mea_end_` and this one.
    void ProcessMeasurement(MeasureGroup& measures);

    FastLioState x;
    Vec3 gravity = Vec3(0, 0, GRAVITY_NORM);  // the sign of RESPLE_LIO::gravity: minus the acceleration of gravity
    int64_t last_end_ns = 0;
    Eigen::aligned_vector<FastLioPoint> points;  // of the last frame
    std::shared_ptr<LaserMappingParam> param;
    // Local BA; its voxel map is the local map the EKF matches against.
    std::unique_ptr<LocalBA> lba;

    // LIGO: the RINEX observations and ephemerides (LaserMapping::ReadObsNav), and the GNSS state.
    obs_t obss{};
    nav_t nav{};
    RTKState rtk_state;

   private:
    // State of the propagation at an IMU sample, with what holds until the next one.
    struct ImuPose {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int64_t time_ns = 0;
        Mat3 R = Mat3::Identity();
        Vec3 p = Vec3::Zero();
        Vec3 v = Vec3::Zero();
        Vec3 gyro = Vec3::Zero();  // bias free
        Vec3 acc = Vec3::Zero();   // world frame, with gravity
    };

    // Voxel-SLAM IMUEKF::motion_blur: x to `end_ns`, and the frame's points to that instant, LiDAR frame.
    void Propagate(const MeasureGroup& measures, int64_t end_ns, std::vector<Vec3>* scan);
    std::vector<Vec3> UniformSample(const std::vector<Vec3>& scan, double size) const;
    // Indices of the points UniformSample keeps.
    std::vector<size_t> UniformSampleIndices(const std::vector<Vec3>& scan, double size) const;
    // FR-LIO Eq. 1: how many sub-frames this frame is cut into, from the spread of its IMU samples.
    int SubFrameCount(const MeasureGroup& measures) const;
    // FR-LIO: n updates on slices of the frame, a backward smoother over their states, and the frame's points
    // re-expressed at the frame end with the smoothed poses.  Leaves x at the frame end and `points` as one frame.
    void ProcessSubFrames(MeasureGroup& measures, int n);
    void SetPoints(const std::vector<Vec3>& scan);
    int Match();
    // Voxel-SLAM lio_state_estimation: LIO on the nav state, or, once the GNSS is aligned, on the nav and GNSS states
    // together with the valid TDCP rows in tdcp_ when there are any (LIGO).  False when the matched planes leave a
    // direction unobserved.
    bool IteratedUpdate();

    // GNSS (LIGO), after RESPLE_LIO::ProcessGnssEpoch, PrepareTdCpEpoch and InitGnssState.
    // Range from the antenna at the IMU pose (R, p) to `sv_pos`; H, when given, is d(range)/d(nav error, GNSS state)
    // with the nav columns only when `nav_cols`.
    double AntennaRange(const Mat3& R, const Vec3& p, const Vec3& sv_pos, bool nav_cols,
                        Eigen::Matrix<double, 1, FastLioState::DIM + RTKState::DIM>* H) const;
    // TDCP residual (measured minus predicted) of one row at the current state, and its Jacobian.
    double TdcpResidual(const TdcpObs& obs, Eigen::Matrix<double, 1, FastLioState::DIM + RTKState::DIM>* H) const;
    // The epoch's clock step from the median clock-free residual of tdcp_epoch_, then the chi-square test on the
    // prior; tdcp_ gets the rows that pass, or stays empty below tdcp_min_sat.  The state must be at the epoch.
    void PrepareTdcp();
    void InitGnss(const sol_t& sol);

    Imu last_imu_;
    int degrade_cnt_ = 0;
    std::unordered_map<int, SatTrack> sat_tracks_;
    std::vector<TdcpObs> tdcp_epoch_;  // this frame's TDCP rows, epoch at the frame end
    std::vector<TdcpObs> tdcp_;        // the valid ones, set by PrepareTdcp just before the frame-end update

    FastLioState::MatN phi_ = FastLioState::MatN::Identity();  // error-state transition of the last Propagate
};

}  // namespace kuma_lio
