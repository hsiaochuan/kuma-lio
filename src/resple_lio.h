#pragma once
#include <glog/logging.h>
#include <array>
#include <pcl/common/transforms.h>
#include <pcl/filters/uniform_sampling.h>

#include "common_lib.h"
#include "eigen_type.h"
#include "ikd-Tree/ikd_Tree.h"
#include "laser_mapping_param.h"
#include "spline_state.h"
#include "timer.h"
namespace faster_lio {
struct PointData {
    Vec3 point_body;
    Vec3 point_world;
    faster_lio::Point point;
    int64_t time_ns;
    Vec3 ppl_normal;
    double ppl_d;
    double z;
    Eigen::aligned_vector<faster_lio::Point> near_points;
    Eigen::Matrix<double, 1, 24> H;
    bool if_valid = false;
    double pt_wt = 0.01;
};
struct ImuData {
    int64_t time_ns;
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
    Eigen::Matrix<double, 6, 30> H;
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
struct TimeDiffCarrierPhaseData {
    int64_t time_ns;
    Vec3 sv_pos;
    double sv_dt;
    Vec3 pos;
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
    static constexpr int XSIZE = 30;
    static constexpr int CP_SIZE = 24;
    static constexpr int S1 = 0, S2 = 6, S3 = 12, S4 = 18;
    static constexpr int Q1 = 3, Q2 = 9, Q3 = 15, Q4 = 21;
    static constexpr int BA_OFFSET = 24;
    static constexpr int BG_OFFSET = 27;
    static constexpr double GRAVITY_NORM = 9.81;
    static constexpr int MATCH_POINT = 5;
    static constexpr int kMinParallelPoints = 2048;
    void Init();

    void ProcessMeasurement(MeasureGroup & measures);

    Eigen::Matrix<double, XSIZE, 1> getState();
    static bool esti_plane(Vec4& pca_result, const Eigen::aligned_vector<faster_lio::Point>& point, const double& threshold);
    int Match();
    void IterativeUpdate();
    void prepTdCp(TimeDiffCarrierPhaseData & tdcp_data);
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

    // Slide the local map cube with the sensor and delete what falls out, ported
    // from RESPLE::lasermapFovSegment.
    void LaserMapFovSegment();

    void mapIncremental();
    bool if_inertial_initialized = false;
    bool if_localmap_initialized = false;
    bool localmap_box_initialized = false;

    bool if_GNSS_aligned = false;

    BoxPointType localmap_box;
    SplineState spl;
    Eigen::Matrix<double, XSIZE, XSIZE> cov_rcp;
    Eigen::Matrix<double, XSIZE, XSIZE> cov_sys;
    Eigen::Matrix<double, XSIZE, XSIZE> a_mat;
    Eigen::Matrix<double, XSIZE, XSIZE> KH;
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, GRAVITY_NORM);
    std::shared_ptr<LaserMappingParam> param;
    std::shared_ptr<KD_TREE<Point>> kd_tree;
    PointCloud accum_points;
    std::vector<Eigen::aligned_vector<faster_lio::Point>> accum_nearest_points;
    Eigen::aligned_deque<ImuData> imu_meas;
    Eigen::aligned_deque<PointData> pt_meas;
    Eigen::aligned_deque<TimeDiffCarrierPhaseData> tdcp_meas;
    Eigen::aligned_deque<PsrData> psr_meas;
    obs_t obss{};
    nav_t nav{};
    std::unordered_map<int, SatTrack> sat_tracks_;
    Mat3 R_ecef_enu;
    Vec3 t_ecef_enu;
};
}  // namespace faster_lio
