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
#include "loop_closure.h"

namespace kuma_lio {

// Voxel-SLAM IMUST: the IMU frame in the world, with a fixed gravity outside the state.
struct FastLioState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int DIM = 15;  // R p v bg ba
    static constexpr int ROT = 0, POS = 3, VEL = 6, BG = 9, BA = 12;
    using MatN = Eigen::Matrix<double, DIM, DIM>;
    using VecN = Eigen::Matrix<double, DIM, 1>;

    FastLioState() {
        cov = MatN::Identity() * 1e-4;
        cov.block<6, 6>(BG, BG) = Eigen::Matrix<double, 6, 6>::Identity() * 1e-5;  // bg and ba
    }
    FastLioState& operator+=(const VecN& dx) {
        R = R * ExpMat(Vec3(dx.segment<3>(ROT)));
        p += dx.segment<3>(POS);
        v += dx.segment<3>(VEL);
        bg += dx.segment<3>(BG);
        ba += dx.segment<3>(BA);
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
        dx.segment<3>(ROT) = LogMat(dR);
        dx.segment<3>(POS) = p - b.p;
        dx.segment<3>(VEL) = v - b.v;
        dx.segment<3>(BG) = bg - b.bg;
        dx.segment<3>(BA) = ba - b.ba;
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
    static constexpr int CLK = 0, YAW = 4, T_ECEF_ENU = 5;
    using Vec = Eigen::Matrix<double, DIM, 1>;

    Mat3 RotEcefEnu() const { return R_ecef_enu_rollpitch * Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(); }

    Vec4 clk = Vec4::Zero();         // c * receiver clock [m]: GPS, GLO, GAL, BDS
    Vec4 clk_anchor = Vec4::Zero();  // the clocks at the anchor epoch, held fixed
    Vec3 t_ecef_enu = Vec3::Zero();
    Mat3 R_ecef_enu_rollpitch = Mat3::Identity();
    double yaw = 0.0;
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

    // `pgo_config`: config/pgo.yaml to run the loop closure coupled to the front end; empty = none.
    void Init(const std::string& pgo_config);
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
    // Loop closure coupled to the BA and the map (--pgo_config); null without.
    std::unique_ptr<LoopClosure> loop;

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
    double AntennaRange(const Mat3& R, const Vec3& pos, const Vec3& sv_pos, bool if_eval_jaco,
                        Eigen::Matrix<double, 1, FastLioState::DIM + RTKState::DIM>* H) const;
    // TDCP residual (measured minus predicted) of one row at the current state, and its Jacobian.
    double TdcpResidual(const TdcpObs& tdcp_obs, Eigen::Matrix<double, 1, FastLioState::DIM + RTKState::DIM>* H) const;
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
