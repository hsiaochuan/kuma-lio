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

#include "resple_lio.h"

#include <ros/param.h>

namespace kuma_lio {

void RESPLE_LIO::Init() {
    const double dt_s = param->resple_.dt_ns * 1e-9;
    // set P0
    P.setZero();
    P.topLeftCorner<CP_SIZE, CP_SIZE>() =
        param->resple_.cov_P0 * dt_s * dt_s * Eigen::Matrix<double, CP_SIZE, CP_SIZE>::Identity();
    P.block<3, 3>(BA_OFFSET, BA_OFFSET) = param->resple_.cov_ba.asDiagonal();
    P.block<3, 3>(BG_OFFSET, BG_OFFSET) = param->resple_.cov_bg.asDiagonal();

    // set Q: each RCP is a (position, orientation increment) pair with its own random walk
    auto q_block = [dt_s](double std_pos, double std_ort) {
        Eigen::Matrix<double, 6, 6> q = Eigen::Matrix<double, 6, 6>::Zero();
        q.topLeftCorner<3, 3>().diagonal().setConstant(std_pos * std_pos * dt_s * dt_s);
        q.bottomRightCorner<3, 3>().diagonal().setConstant(std_ort * std_ort * dt_s * dt_s);
        return q;
    };
    Q_mat = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
    const Eigen::Matrix<double, 6, 6> q_old = q_block(param->resple_.q_std_pos_old, param->resple_.q_std_ort_old);
    for (int i = 0; i < 3; ++i) Q_mat.block<6, 6>(6 * i, 6 * i) = q_old;
    Q_mat.block<6, 6>(18, 18) = q_block(param->resple_.q_std_pos_new, param->resple_.q_std_ort_new);
    Q_mat.block<3, 3>(BA_OFFSET, BA_OFFSET).diagonal().setConstant(param->resple_.q_std_ba * param->resple_.q_std_ba * dt_s);
    Q_mat.block<3, 3>(BG_OFFSET, BG_OFFSET).diagonal().setConstant(param->resple_.q_std_bg * param->resple_.q_std_bg * dt_s);

    // A
    A_mat = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
    Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
    A_mat.block(0, 6, 6, 6) = I6;
    A_mat.block(6, 12, 6, 6) = I6;
    A_mat.block(12, 18, 6, 6) = I6;
    A_mat.block(18, 0, 3, 3) = -Eigen::Matrix3d::Identity();
    A_mat.block(18, 12, 3, 3) = 2 * Eigen::Matrix3d::Identity();
    A_mat.block(21, 9, 3, 3) = Eigen::Matrix3d::Identity();
    A_mat.block(BA_OFFSET, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
    A_mat.block(BG_OFFSET, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();

    param->lba_.num_threads = param->num_threads;
    lba = std::make_unique<LocalBA>(param->lba_);
    LOG(INFO) << "local BA map shared with the odometry: voxel " << param->lba_.voxel_size << " window "
              << param->lba_.win_size;
}
void RESPLE_LIO::ProcessMeasurement(MeasureGroup& measures) {
    Timer timer;
    for (const Imu& imu : measures.imu_) lba->AddImu(imu.time_ns, imu.angular_velocity, imu.linear_acceleration);
    if (measures.lidar_->empty()) {
        LOG(INFO) << "No LiDAR data, skip process measurement";
        return;
    }
    // transform point
    PointCloud::Ptr points_body(new PointCloud);
    pcl::transformPointCloud(*measures.lidar_, *points_body, param->extrin_il_.Mat4d(), true);

    // downsample point
    if (param->resple_.ds_scan_voxel > 0) {
        pcl::UniformSampling<Point> scan_sampler;
        scan_sampler.setRadiusSearch(param->resple_.ds_scan_voxel);
        scan_sampler.setInputCloud(points_body);
        scan_sampler.filter(*points_body);
        std::sort(points_body->points.begin(), points_body->points.end(),
                  [](const Point& p1, const Point& p2) { return p1.GetTimeNs() < p2.GetTimeNs(); });
    }

    // add to imu
    imu_meas.clear();
    imu_meas.resize(measures.imu_.size());
    for (size_t i = 0; i < measures.imu_.size(); ++i) {
        imu_meas[i] =
            ImuData(measures.imu_[i].time_ns, measures.imu_[i].angular_velocity, measures.imu_[i].linear_acceleration);
    }

    // add to lidar points
    pt_meas.clear();
    pt_meas.resize(points_body->size());
    for (int i = 0; i < points_body->size(); ++i) {
        pt_meas[i].point_body = points_body->at(i).getVector3fMap().cast<double>();
        pt_meas[i].point = points_body->at(i);
        pt_meas[i].time_ns = points_body->at(i).GetTimeNs();
        pt_meas[i].pt_wt = param->resple_.w_pt;
        pt_meas[i].body_cov = BodyPointCov(pt_meas[i].point_body, param->lba_.range_err, param->lba_.angle_err);
    }

    // lidar-inertial update
    IterativeUpdate();

    const int num_pt = static_cast<int>(pt_meas.size());
#pragma omp parallel for num_threads(param->num_threads) if (num_pt > kMinParallelPoints)
    for (int i = 0; i < num_pt; ++i) {
        PointData& pt_data = pt_meas[i];
        Vec3 interp_pos = spl.itpPosition(pt_data.time_ns);
        Eigen::Quaterniond interp_q;
        spl.itpQuaternion(pt_data.time_ns, &interp_q);
        pt_data.point_world = interp_q * pt_data.point_body + interp_pos;
    }
    const double t_update = timer.ElapsedMiniSeconds(true);
    UpdateLocalBA(measures);
    const double ba_time = timer.ElapsedMicroSeconds(true);
    LOG(INFO) << "time [ms]: update " << t_update << " lba " << ba_time
              << " raw points " << measures.lidar_->size();
    if (spl.num_knot % 1000 == 0) {
        LOG(INFO) << "Biases at " << NS_TO_S(spl.maxTimeNs()) << " s: ba " << ba.transpose() << " (std "
                  << P.block<3, 3>(BA_OFFSET, BA_OFFSET).diagonal().cwiseSqrt().transpose() << ") bg "
                  << bg.transpose();
    }
}

Eigen::Matrix<double, XSIZE, 1> RESPLE_LIO::getState() {
    Eigen::Matrix<double, CP_SIZE, 1> cps_win = spl.getRCPs();
    Eigen::Matrix<double, XSIZE, 1> state;
    state << cps_win, ba, bg;
    return state;
}

int RESPLE_LIO::Match() {
    const int num_pt = static_cast<int>(pt_meas.size());
#pragma omp parallel for num_threads(param->num_threads) schedule(dynamic) if (num_pt > kMinParallelPoints)
    for (int i = 0; i < num_pt; i++) {
        PointData& pt_data = pt_meas[i];
        pt_data.if_valid = false;
        // interp to point world
        Vec3 interp_pos = spl.itpPosition(pt_data.time_ns);
        Eigen::Quaterniond interp_q;
        spl.itpQuaternion(pt_data.time_ns, &interp_q);
        pt_data.point_world = interp_q * pt_data.point_body + interp_pos;
        const Mat3 R = interp_q.toRotationMatrix();
        const Mat3 var_w = R * pt_data.body_cov * R.transpose() + param->lba_.pose_var * Mat3::Identity();
        PlaneHit hit;
        if (lba->Query(LbaPoint{pt_data.point_world, var_w}, &hit) &&
            pt_data.point_body.norm() > 81.0 * hit.signed_dist * hit.signed_dist) {
            pt_data.if_valid = true;
            pt_data.ppl_normal = hit.normal;
            pt_data.ppl_d = hit.d;
            pt_data.pt_wt = param->resple_.cov_en
                                ? param->resple_.r_base + hit.plane_sigma + hit.normal.dot(var_w * hit.normal)
                                : param->resple_.w_pt;
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

void RESPLE_LIO::IterativeUpdate() {
    const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = P;
    Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
    bool converged = true;
    int num_eff = 0;
    int t = 0;
    const int max_iter = 5;
    const int n_iter = 1;
    const double eps = 0.1;
    Timer timer, stage;
    double t_match = 0.0, t_solve = 0.0;
    for (int i = 0; i < max_iter; i++) {
        Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
        if (converged) {
            stage.ElapsedMiniSeconds(true);
            num_eff = Match();
            t_match += stage.ElapsedMiniSeconds(true);
        }
        if (num_eff > 0) {
            stage.ElapsedMiniSeconds(true);
            UpdateLiDARInertial(num_eff, rcp_prop, cov_prop);
            t_solve += stage.ElapsedMiniSeconds(true);
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
        if (t > n_iter || (i == max_iter - 1)) {
            P = (Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;
            P = 0.5 * (P + P.transpose());
            break;
        }
    }  // iter
    double update_time = timer.ElapsedMiniSeconds();
    LOG(INFO) << "Points: " << pt_meas.size()
              << " Imu: " << imu_meas.size()
              << " Effect Points: " << num_eff
              << " Time: " << update_time << " match " << t_match << " solve " << t_solve;
}
void RESPLE_LIO::prepIMU(ImuData& imu_data) {
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
    kuma_lio::dRotInv(a_w, q_itp, drot);
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
        imu_data.if_acc_valid[i] = std::abs(imu_data.accel(i) - imu_data.imu_itp(i)) <= param->resple_.imu_acc_outlier;
        if (!imu_data.if_acc_valid[i]) {
            imu_data.H.row(i).setZero();
            imu_data.z(i) = 0;
        }
        imu_data.if_gyro_valid[i] = std::abs(imu_data.gyro(i) - imu_data.imu_itp(i + 3)) <= param->resple_.imu_gyro_outlier;
        if (!imu_data.if_gyro_valid[i]) {
            imu_data.H.row(i + 3).setZero();
            imu_data.z(i + 3) = 0;
        }
    }
}

void RESPLE_LIO::prepLiDAR(PointData& pt_data) const {
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
        dRot(pt_data.point_body, q_itp, drot);
        Eigen::Matrix<double, 1, 4> tmp = pt_data.ppl_normal.transpose() * drot;
        for (int i = 0; i < 4; i++) {
            Hi.block(0, i * 6, 1, 3) = pt_data.ppl_normal.transpose() * J_pos.d_val_d_knot[i];
            Hi.block(0, i * 6 + 3, 1, 3) = tmp * J_ortdel.d_val_d_knot[i];
        }
        pt_data.H = Hi.leftCols<24>();
    }
}

void RESPLE_LIO::propRCP() {
    Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
    Eigen::Matrix<double, 6, 1> cp_prop_pos = 2 * cps_win.block<6, 1>(12, 0) - cps_win.block<6, 1>(0, 0);
    Eigen::Vector3d delta = cps_win.segment<3>(9);
    spl.addOneStateKnot(cp_prop_pos.head<3>(), delta);
    P = A_mat * P * A_mat.transpose() + Q_mat;
}

void RESPLE_LIO::UpdateLiDARInertial(int eff_lidar_num, const Eigen::Matrix<double, XSIZE, 1>& x_prop,
                                     const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop) {
    Eigen::Matrix<double, 6, 1> cov_imu_inv;
    cov_imu_inv(0) = 1. / param->resple_.cov_acc(0);
    cov_imu_inv(1) = 1. / param->resple_.cov_acc(1);
    cov_imu_inv(2) = 1. / param->resple_.cov_acc(2);
    cov_imu_inv(3) = 1. / param->resple_.cov_gyro(0);
    cov_imu_inv(4) = 1. / param->resple_.cov_gyro(1);
    cov_imu_inv(5) = 1. / param->resple_.cov_gyro(2);

    const int num_pt = static_cast<int>(pt_meas.size());
#pragma omp parallel for num_threads(param->num_threads) schedule(dynamic) if (num_pt > kMinParallelPoints)
    for (int i = 0; i < num_pt; i++) {
        PointData& pt_data = pt_meas[i];
        prepLiDAR(pt_data);
    }
    for (size_t i = 0; i < imu_meas.size(); i++) {
        prepIMU(imu_meas[i]);
    }
    int dim_meas = 6 * imu_meas.size() + eff_lidar_num;
    Eigen::Matrix<double, Eigen::Dynamic, XSIZE> H(dim_meas, XSIZE);
    Eigen::Matrix<double, Eigen::Dynamic, 1> z(dim_meas, 1);
    Eigen::Matrix<double, Eigen::Dynamic, 1> mat_cov_inv(dim_meas, 1);
    H.setZero();
    z.setZero();
    mat_cov_inv.setZero();
    int idx_offset = 0;
    for (const PointData& pt_data : pt_meas) {
        if (!pt_data.if_valid) continue;
        double lid_cov = pt_data.H * P.topLeftCorner<24, 24>() * pt_data.H.transpose() + pt_data.pt_wt;
        if (std::abs(pt_data.z) < param->resple_.ppl_thr || lid_cov < pt_data.pt_wt * param->resple_.coeff_cov) {
            z(idx_offset) = -pt_data.z;
            H.block(idx_offset, 0, 1, 24) = pt_data.H;
        }
        mat_cov_inv(idx_offset) = 1 / pt_data.pt_wt;
        idx_offset++;
    }
    for (const ImuData& imu_data : imu_meas) {
        z.segment<6>(idx_offset) = imu_data.z;
        H.block(idx_offset, 0, 6, XSIZE) = imu_data.H;
        mat_cov_inv.segment<6>(idx_offset) = cov_imu_inv;
        idx_offset += 6;
    }
    Update(z, mat_cov_inv, H, x_prop, P_prop);
}

void RESPLE_LIO::UpdateLocalBA(const MeasureGroup& measures) {
    for (const Point& pt : measures.lidar_->points) {
        const Vec3 p_body = param->extrin_il_ * Vec3(pt.getVector3fMap().cast<double>());
        Eigen::Quaterniond q;
        spl.itpQuaternion(pt.GetTimeNs(), &q);
        const Mat3 R = q.toRotationMatrix();
        LbaPoint pv;
        pv.p = R * p_body + spl.itpPosition(pt.GetTimeNs());
        pv.var = R * BodyPointCov(p_body, param->lba_.range_err, param->lba_.angle_err) * R.transpose();
        lba_points_.emplace_back(pt.GetTimeNs(), pv);
    }
    // The spline has no frames, so one closes every frame_dt at the start of this knot interval: the
    // points since the last cut, moved into the body frame at this instant.
    const int64_t t_ns = spl.maxTimeNs() - param->resple_.dt_ns;
    if (lba_last_cut_ns_ < 0) lba_last_cut_ns_ = t_ns;
    if (t_ns - lba_last_cut_ns_ < S_TO_NS(param->lba_.frame_dt)) return;
    lba_last_cut_ns_ = t_ns;
    const Eigen::Isometry3d pose = spl.Interpolate(t_ns);
    LbaState odom;
    odom.R = pose.linear();
    odom.p = pose.translation();
    odom.v = spl.itpPosition<1>(t_ns);
    odom.bg = bg;
    odom.ba = ba;
    std::vector<LbaPoint> points;
    while (!lba_points_.empty() && lba_points_.front().first <= t_ns) {
        const LbaPoint& pv = lba_points_.front().second;
        points.push_back(LbaPoint{odom.R.transpose() * (pv.p - odom.p),
                                  odom.R.transpose() * pv.var * odom.R});
        lba_points_.pop_front();
    }
    // local ba
    if (lba->AddFrame(t_ns, odom, points, gravity)) {
        // align the front end
        AlignToBA(t_ns, lba->Newest());
    }
}

void RESPLE_LIO::AlignToBA(int64_t t_ns, const LbaState& x) {
    // One update with both measurements of the refined frame.  Rows: position, then the three body
    // axes in the world frame (they carry the rotation with the Jacobian the LiDAR rows already use;
    // their 9 rows hold 2 / sigma^2 per angle), then the gyroscope and accelerometer biases.
    const RespleParam& lp = param->resple_;
    Eigen::Matrix<double, 18, XSIZE> H = Eigen::Matrix<double, 18, XSIZE>::Zero();
    Eigen::Matrix<double, 18, 1> innov, R_inv;
    Eigen::Quaterniond q;
    Jacobian43 J_ortdel;
    Jacobian J_pos;
    spl.itpQuaternion(t_ns, &q, nullptr, &J_ortdel);
    const Vec3 p = spl.itpPosition(t_ns, &J_pos);
    const Mat3 R = q.toRotationMatrix();
    innov.head<3>() = x.p - p;
    const int recur_st_id = spl.num_knot - 4;
    for (int k = 0; k < 3; ++k) {
        innov.segment<3>(3 + 3 * k) = x.R.col(k) - R.col(k);
        Eigen::Matrix<double, 3, 4> drot;
        dRot(Vec3(Vec3::Unit(k)), q, drot);
        for (int i = 0; i < 4; ++i) {
            const int j = static_cast<int>(J_ortdel.start_idx) + i - recur_st_id;
            if (j < 0) continue;
            H.block<3, 3>(3 + 3 * k, j * 6 + 3) = drot * J_ortdel.d_val_d_knot[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        const int j = static_cast<int>(J_pos.start_idx) + i - recur_st_id;
        if (j < 0) continue;
        H.block<3, 3>(0, j * 6) = J_pos.d_val_d_knot[i] * Mat3::Identity();
    }
    R_inv.head<3>().setConstant(1.0 / (lp.fb_std_pos * lp.fb_std_pos));
    R_inv.segment<9>(3).setConstant(0.5 / (lp.fb_std_ort * lp.fb_std_ort));
    // the newest frame's biases as the BA left them
    H.block<3, 3>(12, BG_OFFSET).setIdentity();
    H.block<3, 3>(15, BA_OFFSET).setIdentity();
    innov.segment<3>(12) = x.bg - bg;
    innov.segment<3>(15) = x.ba - ba;
    R_inv.segment<3>(12).setConstant(1.0 / (lp.fb_std_bg * lp.fb_std_bg));
    R_inv.segment<3>(15).setConstant(1.0 / (lp.fb_std_ba * lp.fb_std_ba));

    const Eigen::Matrix<double, XSIZE, XSIZE> cov_prior = P;
    const Eigen::Matrix<double, XSIZE, 1> x_prior = getState();
    Update<18>(innov, R_inv, H, x_prior, cov_prior);
    if (!lp.fb_keep_cov) {
        P = (Eigen::Matrix<double, XSIZE, XSIZE>::Identity() - KH) * cov_prior;
        P = 0.5 * (P + P.transpose());
    }
    const Eigen::Isometry3d after = spl.Interpolate(t_ns);
    LOG(INFO) << "LBA align: offset " << innov.head<3>().norm() << " m " << innov.segment<9>(3).norm() / std::sqrt(2.0)
              << " rad, left " << (x.p - after.translation()).norm() << " m, bias offset bg "
              << innov.segment<3>(12).norm() << " ba " << innov.segment<3>(15).norm();
}

void RESPLE_LIO::UpdateState(const Eigen::Matrix<double, XSIZE, 1>& xupd) {
    Eigen::Matrix<double, CP_SIZE, 1> cp_win = xupd.segment(0, CP_SIZE);
    for (int i = 0; i < 4; ++i) {
        int knot_id = spl.num_knot - 4 + i;
        CHECK(knot_id >= 0);
        spl.t_knots[knot_id] = cp_win.segment<3>(i * 6);
        spl.ort_delta[knot_id] = cp_win.segment<3>(i * 6 + 3);
        Eigen::Quaterniond q_del = kuma_lio::ExpQuatHalf(spl.ort_delta[knot_id]);
        if (knot_id == 0) {
            spl.q_knots[knot_id] = spl.q_idle * q_del;
        } else {
            spl.q_knots[knot_id] = spl.q_knots[knot_id - 1] * q_del;
        }
    }
    ba = xupd.segment(BA_OFFSET, 3);
    bg = xupd.segment(BG_OFFSET, 3);
}

}  // namespace kuma_lio
