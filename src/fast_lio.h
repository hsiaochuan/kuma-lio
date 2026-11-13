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

// Voxel-SLAM IMUST: the IMU frame in the world, with a fixed gravity outside the state.  LIGO adds the placement of
// the local frame in ECEF, ecef = R_ecef_enu Rz(yaw) local + t_ecef_enu (R_ecef_enu outside the state), and the
// receiver clock of each system [m] (FAST-LIVGO eq. 1, without the drift).  They stay untouched until the front end's
// own alignment places the local frame (FastLio::AlignGnss); the TDCP update estimates them from then on.
struct FastLioState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int DIM = 23;  // R p v bg ba | yaw t_ecef_enu clk(GPS GLO GAL BDS)
    static constexpr int ROT = 0, POS = 3, VEL = 6, BG = 9, BA = 12, YAW = 15, TEC = 16, CLK = 19;
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
        yaw += dx(YAW);
        t_ecef_enu += dx.segment<3>(TEC);
        clk += dx.segment<4>(CLK);
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
        dx(YAW) = yaw - b.yaw;
        dx.segment<3>(TEC) = t_ecef_enu - b.t_ecef_enu;
        dx.segment<4>(CLK) = clk - b.clk;
        return dx;
    }

    Mat3 R = Mat3::Identity();
    Vec3 p = Vec3::Zero();
    Vec3 v = Vec3::Zero();
    Vec3 bg = Vec3::Zero();
    Vec3 ba = Vec3::Zero();
    double yaw = 0.0;                // [rad] about the ENU up axis
    Vec3 t_ecef_enu = Vec3::Zero();  // [m]
    Vec4 clk = Vec4::Zero();         // [m] c * receiver clock bias as read by GPS, GLO, GAL, BDS
    MatN cov;
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

    // LIGO: the RINEX observations and ephemerides (LaserMapping::ReadObsNav).
    obs_t obss{};
    nav_t nav{};

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

    // LIGO, FAST-LIVGO Sec. IV-D-1: the per-satellite TDCP rows of the epoch that closes this frame, against the
    // previous frame end as a fixed anchor (eq. 11): its antenna position in the local frame and its receiver clocks
    // are constants, while yaw and t_ecef_enu, being parameters rather than per-epoch states, are the current ones
    // at both epochs.
    struct Tdcp {
        bool valid = false;
        std::vector<TdcpObs> obs;       // anchored at the previous frame end
        Vec3 ant_anchor = Vec3::Zero();  // antenna at the previous frame end, local frame
        Vec4 clk_anchor = Vec4::Zero();  // receiver clocks at the previous frame end [m]
        std::vector<bool> mask;           // rows past the gate of the first iteration
        int num_used = 0;
        int num_gated = 0;
        double postfit_rms = 0.0;        // [m] over the rows used, at the updated state
        double postfit_nrms = 0.0;       // the same, each row over its sigma
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
    // Voxel-SLAM lio_state_estimation on the nav state; in LIGO mode the TDCP rows of the frame end (tdcp_) join
    // the LiDAR rows.  False when the matched planes leave a direction unobserved.
    bool IteratedUpdate();

    // Per row of tdcp_, predicted minus measured TDCP at `s` [m] (eq. 11) and, when `jac` is given, its Jacobian
    // on the error state.
    void TdcpResiduals(const FastLioState& x, std::vector<double>* residuals,
                       Eigen::aligned_vector<FastLioState::VecN>* jacos) const;
    // The TDCP rows into the normal equations of IteratedUpdate; `if_gate` (the first iteration) decides which rows
    // are used: the chi-square test on the prior (eq. 14).
    void AddTdcpObservations(bool if_gate, const FastLioState& x_prop, FastLioState::MatN* HTH, FastLioState::VecN* HTz);
    // LIGO, until aligned, with the epoch that closed this frame (x at its end): the two-step placement of the local
    // frame in ECEF, R_ecef_enu from the first SPP fix, then yaw and t_ecef_enu from GnssAlignment once the antenna
    // track has spread ali_min_travel; with ali_refine the yaw is re-estimated from the window's TDCP
    // (RefineYawTdcp) and t_ecef_enu re-matched to the SPP centroids under it.  Turns the GNSS states on, with the
    // SPP clocks and their initial covariance.
    void AlignGnss(const GnssEpoch& epoch);

    Imu last_imu_;
    int degrade_cnt_ = 0;
    std::unordered_map<int, SatTrack> sat_tracks_;
    Tdcp tdcp_;
    bool gnss_aligned_ = false;
    Mat3 R_ecef_enu_ = Mat3::Identity();  // the ENU frame at the first SPP fix
    // the alignment's data until it succeeds: antenna in the local frame and its SPP fix every 0.3 m, and, for
    // ali_refine, every epoch by its time with its antenna
    std::vector<Vec3> align_locals_, align_ecefs_;
    GnssAlignWindow align_window_;

    FastLioState::MatN phi_ = FastLioState::MatN::Identity();  // error-state transition of the last Propagate
};

}  // namespace kuma_lio
