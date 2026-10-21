#pragma once
#include <glog/logging.h>
#include <array>
#include <pcl/common/transforms.h>
#include <pcl/filters/uniform_sampling.h>

#include "common_lib.h"
#include "eigen_type.h"
#include "laser_mapping_param.h"
#include "spline_state.h"
#include "timer.h"
namespace faster_lio {
// window control points (4 x 6) | ba | bg | receiver clocks (4) | yaw_ecef_enu (1) | t_ecef_enu
constexpr int XSIZE = 38;
struct PointData {
    Vec3 point_body;
    Vec3 point_world;
    faster_lio::Point point;
    int64_t time_ns;
    Vec3 ppl_normal;
    double ppl_d;
    double z;
    Eigen::Matrix<double, 1, 24> H;
    bool if_valid = false;
    double pt_wt = 0.01;
    Mat3 body_cov = Mat3::Zero();  ///< LiDAR return covariance in the body frame
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
// Fixed-anchor TDCP (FAST-LIVGO eq. 11): epoch k is the anchor, epoch k+1 lies in the window.
struct TimeDiffCarrierPhaseData {
    int64_t time_ns;         // epoch k+1
    int64_t anchor_time_ns;  // epoch k
    Vec3 sv_pos;             // satellite ECEF at k+1
    Vec3 sv_pos_anchor;      // satellite ECEF at k
    double sv_dt;            // satellite clock bias at k+1 [s]
    double sv_dt_anchor;     // satellite clock bias at k [s]
    double cp_diff;          // lambda * (L_{k+1} - L_k) [m]
    int sys_idx;             // which receiver clock the satellite reads
    double var;              // measurement variance [m^2]
    Eigen::Matrix<double, 1, XSIZE> H;
    double z;
    bool if_valid = false;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
struct PsrData {
    int64_t time_ns;
    Vec3 sv_pos;
    double sv_dt;
    double ion_delay;
    double tro_delay;
    double noise_var;
    double psr_measure;
};
class RESPLE_LIO {
   public:
    static constexpr int CP_SIZE = 24;
    static constexpr int S1 = 0, S2 = 6, S3 = 12, S4 = 18;
    static constexpr int Q1 = 3, Q2 = 9, Q3 = 15, Q4 = 21;
    static constexpr int BA_OFFSET = 24;
    static constexpr int BG_OFFSET = 27;
    static constexpr int CLK_OFFSET = 30;    // c * receiver clock [m]: GPS, GLO, GAL, BDS
    static constexpr int ROT_OFFSET = 34;    // yaw, R_ecef_enu = R_ecef_enu_init * Rz(yaw), 1-dim
    static constexpr int TRANS_OFFSET = 35;  // t_ecef_enu [m]
    static constexpr int GNSS_SIZE = 8;
    static constexpr double GRAVITY_NORM = 9.81;
    static constexpr int kMinParallelPoints = 2048;
    void Init();

    void ProcessMeasurement(MeasureGroup & measures);

    Eigen::Matrix<double, XSIZE, 1> getState();
    int Match();
    void IterativeUpdate();
    bool ProcessGnssEpoch(int ei, int64_t local_time, sol_t& sol);
    void PrepareTdCpEpoch();
    void InitGnssState(const sol_t& sol);
    Mat3 RotEcefEnu() const { return R_ecef_enu_init * Eigen::AngleAxisd(theta_ecef_enu, Vec3::UnitZ()).toRotationMatrix(); }
    double AntennaRange(int64_t t_ns, const Vec3& sv_pos, Eigen::Matrix<double, 1, XSIZE>* H) const;
    void prepTdCp(TimeDiffCarrierPhaseData& tdcp_data) const;
    void prepPsr(PsrData& psr_data);
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

    bool if_GNSS_aligned = false;

    SplineState spl;
    Eigen::Matrix<double, XSIZE, XSIZE> cov_rcp;
    Eigen::Matrix<double, XSIZE, XSIZE> cov_sys;
    Eigen::Matrix<double, XSIZE, XSIZE> a_mat;
    Eigen::Matrix<double, XSIZE, XSIZE> KH;
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, GRAVITY_NORM);
    std::shared_ptr<LaserMappingParam> param;
    /// Local BA; its voxel map is the local map Match() queries.
    std::unique_ptr<LocalBA> lba;
    /// Registered raw returns (world frame) waiting for the cut that closes their frame.
    std::deque<std::pair<int64_t, LbaPoint>> lba_points_;
    /// Close a LiDAR frame every `frame_dt`,
    /// run the BA and align the spline to it.
    void UpdateLocalBA(const MeasureGroup& measures);
    int64_t lba_last_cut_ns_ = -1;
    /// At rest: the IMU is quiet and the trajectory has not moved over the detector's window.  Then
    /// the scans repeat themselves, so the map takes no points and the velocities are measured as zero.
    StaticDetector static_detector;
    bool is_static = false;
    bool TrajectoryStill();
    void ZeroVelocityUpdate();
    /// EKF update of the window with the refined pose `x` at `t_ns` as a measurement.
    void AlignToBA(int64_t t_ns, const LbaState& x);
    Eigen::aligned_deque<ImuData> imu_meas;
    Eigen::aligned_deque<PointData> pt_meas;
    Eigen::aligned_deque<TimeDiffCarrierPhaseData> tdcp_meas;
    Eigen::aligned_deque<PsrData> psr_meas;
    obs_t obss{};
    nav_t nav{};
    std::unordered_map<int, SatTrack> sat_tracks_;


    Vec4 clk = Vec4::Zero();
    Vec4 clk_anchor = Vec4::Zero();  // receiver clocks at the anchor epoch, held fixed
    Mat3 R_ecef_enu_init = Mat3::Identity();
    double theta_ecef_enu = 0.0;  // yaw-only correction on top of R_ecef_enu_init
    Vec3 t_ecef_enu = Vec3::Zero();

    // SPP track collected for the initial local-to-ECEF alignment
    std::vector<Vec3> align_locals;
    std::vector<Vec3> align_ecefs;
};
}  // namespace faster_lio
