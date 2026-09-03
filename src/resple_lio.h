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
          if_acc_valid(other.if_acc_valid), if_gyro_valid(other.if_gyro_valid) {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
    void Init() {

        const double dt_s = param->dt_ns * 1e-9;
        // set P0
        cov_rcp.setZero();
        cov_rcp.topLeftCorner<CP_SIZE, CP_SIZE>() = param->cov_P0 * dt_s * dt_s * Eigen::Matrix<double, CP_SIZE, CP_SIZE>::Identity();
        cov_rcp.block<3, 3>(BA_OFFSET, BA_OFFSET) = param->cov_ba.asDiagonal();
        cov_rcp.block<3, 3>(BG_OFFSET, BG_OFFSET) = param->cov_bg.asDiagonal();

        // set Q
        double cov_sys_pos = param->std_sys_pos * param->std_sys_pos * dt_s * dt_s;
        double cov_sys_ort = param->std_sys_ort * param->std_sys_ort * dt_s * dt_s;
        Eigen::Matrix<double, XSIZE, XSIZE> Q = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
        Eigen::Matrix<double, 6, 6> Q_block_old = Eigen::Matrix<double, 6, 6>::Zero();
        Q_block_old.topLeftCorner<3, 3>() =
            param->cov_RCP_pos_old * cov_sys_pos * Eigen::Matrix3d::Identity();
        Q_block_old.bottomRightCorner<3, 3>() =
            param->cov_RCP_ort_old * cov_sys_ort * Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 6, 6> Q_block_new = Eigen::Matrix<double, 6, 6>::Zero();
        Q_block_new.topLeftCorner<3, 3>() =
            param->cov_RCP_pos_new * cov_sys_pos * Eigen::Matrix3d::Identity();
        Q_block_new.bottomRightCorner<3, 3>() =
            param->cov_RCP_ort_new * cov_sys_ort * Eigen::Matrix3d::Identity();
        Q.topLeftCorner<6, 6>() = Q_block_old;
        Q.block<6, 6>(6, 6) = Q_block_old;
        Q.block<6, 6>(12, 12) = Q_block_old;
        Q.block<6, 6>(18, 18) = Q_block_new;
        cov_sys = Q;


        // A
        a_mat = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
        Eigen::Matrix<double, 6, 6> matblock = Eigen::Matrix<double, 6, 6>::Zero();
        matblock.topLeftCorner<3, 3>().setIdentity();
        matblock.bottomRightCorner<3, 3>().setIdentity();
        a_mat.block(0, 6, 6, 6) = matblock;
        a_mat.block(6, 12, 6, 6) = matblock;
        a_mat.block(12, 18, 6, 6) = matblock;
        a_mat.block(18, 0, 3, 3) = -Eigen::Matrix3d::Identity();
        a_mat.block(18, 12, 3, 3) = 2 * Eigen::Matrix3d::Identity();
        a_mat.block(21, 9, 3, 3) = Eigen::Matrix3d::Identity();
        a_mat.block(BA_OFFSET, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
        a_mat.block(BG_OFFSET, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();

        // ikd tree
        kd_tree = std::make_shared<KD_TREE<faster_lio::Point>>();
        kd_tree->set_downsample_param(param->ds_lm_voxel);

    }

    void ProcessMeasurement(MeasureGroup & measures) {
        if (measures.lidar_->empty()) {
            LOG(INFO) << "No LiDAR data, skip process measurement";
            return;
        }
        // transform point
        PointCloud::Ptr points_body(new PointCloud);
        pcl::transformPointCloud(*measures.lidar_, *points_body, param->extrin_il_.Mat4d(),true);

        // downsample point
        if (param->ds_scan_voxel > 0) {
            pcl::UniformSampling<Point> scan_sampler;
            scan_sampler.setRadiusSearch(param->ds_scan_voxel);
            scan_sampler.setInputCloud(points_body);
            scan_sampler.filter(*points_body);
            std::sort(points_body->points.begin(), points_body->points.end(),
              [](const Point &p1, const Point &p2) { return p1.GetTimeNs() < p2.GetTimeNs(); });
        }
        imu_meas.clear();
        imu_meas.resize(measures.imu_.size());
        for (size_t i = 0; i < measures.imu_.size(); ++i) {
            imu_meas[i] = ImuData(measures.imu_[i].time_ns,
                measures.imu_[i].angular_velocity,
                measures.imu_[i].linear_acceleration);
        }
        pt_meas.clear();
        pt_meas.resize(points_body->size());
        for (int i = 0; i < points_body->size(); ++i) {
            pt_meas[i].point_body = points_body->at(i).getVector3fMap().cast<double>();
            pt_meas[i].point = points_body->at(i);
            pt_meas[i].time_ns = points_body->at(i).GetTimeNs();
            pt_meas[i].pt_wt = param->w_pt;
        }

        IterativeUpdate();
#pragma omp parallel for num_threads(8)
        for (int i = 0; i < pt_meas.size(); ++i) {
            PointData& pt_data = pt_meas[i];
            Vec3 interp_pos = spl.itpPosition(pt_data.time_ns);
            Eigen::Quaterniond interp_q;
            spl.itpQuaternion(pt_data.time_ns, &interp_q);
            pt_data.point_world = interp_q * pt_data.point_body + interp_pos;
        }
        accum_points.reserve(accum_points.size() + pt_meas.size());
        accum_nearest_points.reserve(accum_nearest_points.size() + pt_meas.size());
        for (int i = 0; i < pt_meas.size(); ++i) {
            faster_lio::Point point{};
            point.getVector3fMap() = pt_meas[i].point_world.cast<float>();
            point.SetTimeNs(pt_meas[i].time_ns);
            accum_points.push_back(point);
            accum_nearest_points.push_back(pt_meas[i].near_points);
        }
        if (accum_points.size() < 2) return;
        int64_t accum_points_interval = accum_points.back().GetTimeNs() - accum_points.front().GetTimeNs();
        if (accum_points_interval > S_TO_NS(0.1)) {
            mapIncremental();
            LaserMapFovSegment();
            LOG(INFO) << "Add Points: " << accum_points.size();
            accum_points.clear();
            accum_nearest_points.clear();
        }
    }

    Eigen::Matrix<double, XSIZE, 1> getState() {
        Eigen::Matrix<double, CP_SIZE, 1> cps_win = spl.getRCPs();
        Eigen::Matrix<double, XSIZE, 1> state;
        state << cps_win, ba, bg;
        return state;
    }
    static bool esti_plane(Vec4& pca_result, const Eigen::aligned_vector<faster_lio::Point>& point,const double& threshold) {
        Eigen::Matrix<double, MATCH_POINT, 3> A;
        Eigen::Matrix<double, MATCH_POINT, 1> b;
        A.setZero();
        b.setOnes();
        b *= -1.0f;
        for (int j = 0; j < MATCH_POINT; j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }
        Eigen::Matrix<double, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
        double n = normvec.norm();
        pca_result(0) = normvec(0) / n;
        pca_result(1) = normvec(1) / n;
        pca_result(2) = normvec(2) / n;
        pca_result(3) = 1.0 / n;
        for (int j = 0; j < MATCH_POINT; j++) {
            if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z +
                     pca_result(3)) > threshold) {
                return false;
            }
        }
        return true;
    }
    int Match() {
        const float nn_radius = static_cast<float>(param->nn_search_radius);
        const float nn_max_sq = static_cast<float>(param->nn_search_radius * param->nn_search_radius);
#pragma omp parallel for num_threads(8) schedule(dynamic)
        for (int i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            pt_data.if_valid = false;
            // interp to point world
            Vec3 interp_pos = spl.itpPosition(pt_data.time_ns);
            Eigen::Quaterniond interp_q;
            spl.itpQuaternion(pt_data.time_ns, &interp_q);
            pt_data.point_world = interp_q * pt_data.point_body + interp_pos;
            faster_lio::Point point_w{};
            point_w.getVector3fMap() = pt_data.point_world.cast<float>();
            // nearest search
            std::vector<float> point_search_dists;
            kd_tree->Nearest_Search(point_w, MATCH_POINT, pt_data.near_points, point_search_dists, nn_radius);
            if (static_cast<int>(pt_data.near_points.size()) >= MATCH_POINT &&
                point_search_dists[MATCH_POINT - 1] < nn_max_sq) {
                Vec4 abcd = Vec4::Zero();
                if (esti_plane(abcd, pt_data.near_points, param->plane_thresh)) {
                    double pd2 = abcd.dot(pt_data.point_world.homogeneous());
                    if (pt_data.point_body.norm() > 81.0 * pd2 * pd2) {
                        pt_data.if_valid = true;
                        pt_data.ppl_normal = abcd.head<3>().normalized();
                        pt_data.ppl_d = abcd(3);
                    }
                }
            }
        }  // for point

        int eff_count = 0;
        for (int i = 0; i < pt_meas.size(); i++) {
            if (pt_meas[i].if_valid) {
                eff_count++;
            }
        }
        return eff_count;
    }
    void IterativeUpdate() {
        const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = cov_rcp;
        Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
        bool converged = true;
        int num_eff = 0;
        int t = 0;
        const int max_iter = 5;
        const int n_iter = 1;
        const double eps = 0.1;
        Timer timer;
        for (int i = 0; i < max_iter; i++) {
            Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
            if (converged) {
                num_eff = Match();
            }
            if (num_eff > 0) {
                UpdateLiDARInertial(num_eff, rcp_prop, cov_prop);
            } else {
                LOG(INFO) << "No effective points";
                break;
            }
            converged = true;
            Eigen::Matrix<double, XSIZE, 1> state_af = getState();
            if ((state_af - rcpi).norm() > eps) {
                converged = false;
            } else {
                t++;
            }
            if (!t && i == max_iter - 2) {
                converged = true;
            }
            if ((t > n_iter) || (i == max_iter - 1)) {
                cov_rcp = (Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;
                cov_rcp = 0.5 * (cov_rcp + cov_rcp.transpose());
                break;
            }
        }  // iter
        double update_time = timer.ElapsedMiniSeconds();
        LOG(INFO) << "Points: " << pt_meas.size() << " Imu: " << imu_meas.size() << " Effect Points: " << num_eff << " Time: " << update_time;
    }
    void prepIMU(ImuData& imu_data) {
        Eigen::Quaterniond q_itp;
        Eigen::Vector3d rot_vel;
        Jacobian43 J_ortdel;
        Jacobian J_line_acc;
        Jacobian33 J_gyro;
        spl.itpQuaternion(imu_data.time_ns, &q_itp, &rot_vel, &J_ortdel, &J_gyro);
        Eigen::Vector3d a_w_no_g = spl.itpPosition<2>(imu_data.time_ns, &J_line_acc);
        Eigen::Vector3d a_w = a_w_no_g + gravity;
        Eigen::Matrix3d RT = q_itp.inverse().toRotationMatrix();
        Eigen::Matrix<double, 3, 4> drot;
        faster_lio::drot(a_w, q_itp, drot);
        Eigen::Matrix<double, 6, XSIZE> Hi = Eigen::Matrix<double, 6, XSIZE>::Zero();
        // map knot Jacobians into the recursive-window columns; measurements in an
        // older segment touch a knot outside the window, whose column is dropped
        const int recur_st_id = spl.num_knot - 4;
        for (int i = 0; i < 4; i++) {
            int j = static_cast<int>(J_line_acc.start_idx) + i - recur_st_id;
            if (j < 0) continue;
            Hi.block(0, j * 6, 3, 3) = RT * J_line_acc.d_val_d_knot[i];
            Hi.block(0, j * 6 + 3, 3, 3) = drot * J_ortdel.d_val_d_knot[i];
            Hi.block(3, j * 6 + 3, 3, 3) = J_gyro.d_val_d_knot[i];
        }
        Hi.block(0, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
        Hi.block(3, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
        imu_data.imu_itp.head<3>() = RT * a_w + ba;
        imu_data.imu_itp.tail<3>() = rot_vel + bg;
        imu_data.H = Hi;

        Eigen::Matrix<double, 6, 1> imu;
        imu.head<3>() = imu_data.accel;
        imu.tail<3>() = imu_data.gyro;
        imu_data.imu_itp.head<3>() = RT * a_w + ba;
        imu_data.imu_itp.tail<3>() = rot_vel + bg;
        imu_data.z = imu - imu_data.imu_itp;

        for (int i = 0; i < 3; i++) {
            imu_data.if_acc_valid[i] = std::abs(imu_data.accel(i) - imu_data.imu_itp(i)) <= param->imu_acc_outlier;
            if (!imu_data.if_acc_valid[i]) {
                imu_data.H.row(i).setZero();
                imu_data.z(i) = 0;
            }
            imu_data.if_gyro_valid[i] =
                std::abs(imu_data.gyro(i) - imu_data.imu_itp(i + 3)) <= param->imu_gyro_outlier;
            if (!imu_data.if_gyro_valid[i]) {
                imu_data.H.row(i + 3).setZero();
                imu_data.z(i + 3) = 0;
            }
        }
    }

    void prepLiDAR(PointData& pt_data) const {
        if (pt_data.if_valid) {
            Eigen::Matrix<double, 1, XSIZE> Hi = Eigen::Matrix<double, 1, XSIZE>::Zero();
            Eigen::Quaterniond q_itp;
            Jacobian43 J_ortdel;
            Jacobian J_pos;
            spl.itpQuaternion(pt_data.time_ns, &q_itp, nullptr, &J_ortdel);
            Eigen::Vector3d p_itp = spl.itpPosition(pt_data.time_ns, &J_pos);
            Eigen::Vector3d pt_w = q_itp * pt_data.point_body + p_itp;
            pt_data.z = pt_data.ppl_normal.dot(pt_w) + pt_data.ppl_d;

            Eigen::Matrix<double, 3, 4> drot;
            drotInv(pt_data.point_body, q_itp, drot);
            Eigen::Matrix<double, 1, 4> tmp = pt_data.ppl_normal.transpose() * drot;
            const int recur_st_id = spl.num_knot - 4;
            for (int i = 0; i < 4; i++) {
                int j = static_cast<int>(J_pos.start_idx) + i - recur_st_id;
                if (j < 0) continue;
                Hi.block(0, j * 6, 1, 3) = pt_data.ppl_normal.transpose() * J_pos.d_val_d_knot[i];
                Hi.block(0, j * 6 + 3, 1, 3) = tmp * J_ortdel.d_val_d_knot[i];
            }
            pt_data.H = Hi.leftCols<24>();
        }
    }
    void propRCP() {
        Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
        Eigen::Matrix<double, 6, 1> cp_prop_pos =
            2 * cps_win.block<6, 1>(12, 0) - cps_win.block<6, 1>(0, 0);
        Eigen::Vector3d delta = cps_win.segment<3>(9);
        spl.addOneStateKnot(cp_prop_pos.head<3>(), delta);
        cov_rcp = a_mat * cov_rcp * a_mat.transpose() + cov_sys;
    }
    void UpdateLiDARInertial(int eff_lidar_num, const Eigen::Matrix<double, XSIZE, 1>& x_prop,
                             const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop) {
        Eigen::Matrix<double, 6, 1> cov_imu_inv;
        cov_imu_inv(0) = 1. / param->cov_acc(0);
        cov_imu_inv(1) = 1. / param->cov_acc(1);
        cov_imu_inv(2) = 1. / param->cov_acc(2);
        cov_imu_inv(3) = 1. / param->cov_gyro(0);
        cov_imu_inv(4) = 1. / param->cov_gyro(1);
        cov_imu_inv(5) = 1. / param->cov_gyro(2);

#pragma omp parallel for num_threads(8) schedule(dynamic)
        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            prepLiDAR(pt_data);
        }
#pragma omp parallel for num_threads(8)
        for (size_t i = 0; i < imu_meas.size(); i++) {
            prepIMU(imu_meas[i]);
        }
        int dim_meas = 6 * imu_meas.size() + eff_lidar_num;
        Eigen::Matrix<double, Eigen::Dynamic, XSIZE> H(dim_meas, XSIZE);
        Eigen::Matrix<double, Eigen::Dynamic, 1> innv(dim_meas, 1);
        Eigen::Matrix<double, Eigen::Dynamic, 1> mat_cov_inv(dim_meas, 1);
        H.setZero();
        innv.setZero();
        mat_cov_inv.setZero();
        int idx_offset = 0;
        size_t id_imu = 0;
        size_t id_pt = 0;

        for (size_t j = 0; j < imu_meas.size() + pt_meas.size(); j++) {
            if ((id_pt < pt_meas.size() && id_imu < imu_meas.size() &&
                 pt_meas[id_pt].time_ns < imu_meas[id_imu].time_ns) ||
                (id_pt < pt_meas.size() && id_imu >= imu_meas.size())) {
                PointData& pt_data = pt_meas[id_pt];
                if (pt_data.if_valid) {
                    double lid_cov = pt_data.H * cov_rcp.topLeftCorner<24, 24>() * pt_data.H.transpose() + pt_data.pt_wt;
                    if (std::abs(pt_data.z) < param->ppl_thr || lid_cov < pt_data.pt_wt * param->coeff_cov) {
                        innv(idx_offset) = -pt_data.z;
                        H.block(idx_offset, 0, 1, 24) = pt_data.H;
                    }
                    mat_cov_inv(idx_offset) = 1 / pt_data.pt_wt;
                    idx_offset++;
                }
                id_pt++;
            } else if ((id_pt < pt_meas.size() && id_imu < imu_meas.size() &&
                        pt_meas[id_pt].time_ns >= imu_meas[id_imu].time_ns) ||
                       (id_pt >= pt_meas.size() && id_imu < imu_meas.size())) {
                const ImuData& imu_data = imu_meas[id_imu];
                innv.segment<6>(idx_offset) = imu_data.z;
                H.block(idx_offset, 0, 6, XSIZE) = imu_data.H;
                mat_cov_inv.segment<6>(idx_offset) = cov_imu_inv;
                idx_offset += 6;
                id_imu++;
            }
        }
        Update(innv, mat_cov_inv, H, x_prop, P_prop);
    }

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
    void UpdateState(const Eigen::Matrix<double, XSIZE, 1>& xupd) {
        Eigen::Matrix<double, CP_SIZE, 1> cp_win = xupd.segment(0, CP_SIZE);
        for (int i = 0; i < 4; ++i) {
            int knot_id = spl.num_knot - 4 + i;
            CHECK(knot_id >= 0);
            spl.t_knots[knot_id] = cp_win.segment<3>(i * 6);
            spl.ort_delta[knot_id] = cp_win.segment<3>(i * 6 + 3);
            Eigen::Quaterniond q_del = faster_lio::ExpQuatHalf(spl.ort_delta[knot_id]);
            if (knot_id == 0) {
                spl.q_knots[knot_id] = spl.q_idle * q_del;
            } else {
                spl.q_knots[knot_id] = spl.q_knots[knot_id - 1] * q_del;
            }
        }
        ba = xupd.segment(BA_OFFSET, 3);
        bg = xupd.segment(BG_OFFSET, 3);
    }

    // Slide the local map cube with the sensor and delete what falls out, ported
    // from RESPLE::lasermapFovSegment.
    void LaserMapFovSegment() {
        Vec3 pos_lidar = spl.itpPosition(spl.maxTimeNs() - 1);
        if (!localmap_box_initialized) {
            for (int i = 0; i < 3; i++) {
                localmap_box.vertex_min[i] = pos_lidar(i) - param->cube_len / 2.0;
                localmap_box.vertex_max[i] = pos_lidar(i) + param->cube_len / 2.0;
            }
            localmap_box_initialized = true;
            return;
        }
        const float mov_threshold = 1.5;
        const float det_range = 100.0;
        float dist_to_map_edge[3][2];
        bool need_move = false;
        for (int i = 0; i < 3; i++) {
            dist_to_map_edge[i][0] = fabs(pos_lidar(i) - localmap_box.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_lidar(i) - localmap_box.vertex_max[i]);
            if (dist_to_map_edge[i][0] <= mov_threshold * det_range ||
                dist_to_map_edge[i][1] <= mov_threshold * det_range)
                need_move = true;
        }
        if (!need_move) return;
        std::vector<BoxPointType> cub_needrm;
        BoxPointType new_localmap_box = localmap_box, tmp_box;
        float mov_dist = std::max((param->cube_len - 2.0 * mov_threshold * det_range) * 0.5 * 0.9,
                                  double(det_range * (mov_threshold - 1)));
        for (int i = 0; i < 3; i++) {
            tmp_box = localmap_box;
            if (dist_to_map_edge[i][0] <= mov_threshold * det_range) {
                new_localmap_box.vertex_max[i] -= mov_dist;
                new_localmap_box.vertex_min[i] -= mov_dist;
                tmp_box.vertex_min[i] = localmap_box.vertex_max[i] - mov_dist;
                cub_needrm.emplace_back(tmp_box);
            } else if (dist_to_map_edge[i][1] <= mov_threshold * det_range) {
                new_localmap_box.vertex_max[i] += mov_dist;
                new_localmap_box.vertex_min[i] += mov_dist;
                tmp_box.vertex_max[i] = localmap_box.vertex_min[i] + mov_dist;
                cub_needrm.emplace_back(tmp_box);
            }
        }
        localmap_box = new_localmap_box;
        if (!cub_needrm.empty()) kd_tree->Delete_Point_Boxes(cub_needrm);
    }

    void mapIncremental() {
        Eigen::aligned_vector<faster_lio::Point> PointToAdd;
        Eigen::aligned_vector<faster_lio::Point> PointNoNeedDownsample;
        int feats_down_size = accum_points.points.size();
        PointToAdd.reserve(feats_down_size);
        PointNoNeedDownsample.reserve(feats_down_size);
        for (int i = 0; i < feats_down_size; i++) {
            const faster_lio::Point& point = accum_points.points[i];
            if (!accum_nearest_points[i].empty()) {
                Eigen::aligned_vector<faster_lio::Point>& points_near = accum_nearest_points[i];
                bool need_add = true;
                double ds_lm_voxel = param->ds_lm_voxel;
                faster_lio::Point mid_point;
                mid_point.x = floor(point.x / ds_lm_voxel) * ds_lm_voxel + 0.5 * ds_lm_voxel;
                mid_point.y = floor(point.y / ds_lm_voxel) * ds_lm_voxel + 0.5 * ds_lm_voxel;
                mid_point.z = floor(point.z / ds_lm_voxel) * ds_lm_voxel + 0.5 * ds_lm_voxel;
                if (fabs(points_near[0].x - mid_point.x) > 0.866 * ds_lm_voxel ||
                    fabs(points_near[0].y - mid_point.y) > 0.866 * ds_lm_voxel ||
                    fabs(points_near[0].z - mid_point.z) > 0.866 * ds_lm_voxel) {
                    PointNoNeedDownsample.emplace_back(accum_points.points[i]);
                    continue;
                }
                for (size_t readd_i = 0; readd_i < points_near.size(); readd_i++) {
                    if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * ds_lm_voxel &&
                        fabs(points_near[readd_i].y - mid_point.y) < 0.5 * ds_lm_voxel &&
                        fabs(points_near[readd_i].z - mid_point.z) < 0.5 * ds_lm_voxel) {
                        need_add = false;
                        break;
                    }
                }
                if (need_add) PointToAdd.emplace_back(point);
            } else {
                PointNoNeedDownsample.emplace_back(point);
            }
        }
        kd_tree->Add_Points(PointToAdd, true);
        kd_tree->Add_Points(PointNoNeedDownsample, false);
    }
    bool if_inertial_initialized = false;
    bool if_localmap_initialized = false;
    bool localmap_box_initialized = false;
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
};
}  // namespace faster_lio
