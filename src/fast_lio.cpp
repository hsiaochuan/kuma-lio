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

#include "fast_lio.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <unordered_map>

#include "timer.h"

namespace kuma_lio {
namespace {
constexpr int kMinParallelPoints = 2048;
constexpr int kMaxIter = 4;
constexpr int kMinScanPoints = 500;     // below this the frame is downsampled at half the size
constexpr double kMinNormalEigen = 14;  // smallest eigenvalue of sum(n n^T) over a frame's matches
}  // namespace

void FastLio::Init() {
    param->lba_.num_threads = param->num_threads;
    lba = std::make_unique<LocalBA>(param->lba_);
    LOG(INFO) << "local BA map shared with the odometry: voxel " << param->lba_.voxel_size << " window "
              << param->lba_.win_size;
}

void FastLio::Start(const FastLioState& state, const Vec3& gravity_up, int64_t time_ns, const Imu& last_imu) {
    x = state;
    gravity = gravity_up;
    last_end_ns = time_ns;
    last_imu_ = last_imu;
}

void FastLio::Propagate(const MeasureGroup& measures, int64_t end_ns, std::vector<Vec3>* scan) {
    const FastLioParam& fp = param->fast_lio_;
    const Vec3 g = -gravity;
    const Vec3 cov_gyr = Vec3::Constant(fp.cov_gyr);
    const Vec3 cov_acc = Vec3::Constant(fp.cov_acc);
    const Vec3 cov_bg = Vec3::Constant(fp.rdw_gyr);
    const Vec3 cov_ba = Vec3::Constant(fp.rdw_acc);

    std::vector<const Imu*> imus;
    imus.reserve(measures.imu_.size() + 1);
    imus.push_back(&last_imu_);
    for (const Imu& imu : measures.imu_) imus.push_back(&imu);

    Eigen::aligned_vector<ImuPose> poses;
    poses.reserve(imus.size());
    Vec3 acc_w = x.R * (last_imu_.linear_acceleration - x.ba) + g;
    Vec3 gyr = last_imu_.angular_velocity - x.bg;
    Vec3 pos = x.p, vel = x.v;
    Mat3 rot = x.R;
    FastLioState::MatN F, cov_w;
    phi_.setIdentity();
    for (size_t i = 0; i + 1 < imus.size(); ++i) {
        const Imu& head = *imus[i];
        const Imu& tail = *imus[i + 1];
        if (tail.time_ns < last_end_ns) continue;
        gyr = 0.5 * (head.angular_velocity + tail.angular_velocity) - x.bg;
        const Vec3 acc = 0.5 * (head.linear_acceleration + tail.linear_acceleration) - x.ba;
        acc_w = rot * acc + g;

        const int64_t cur_ns = std::max(head.time_ns, last_end_ns);
        const double dt = NS_TO_S(tail.time_ns - cur_ns);
        ImuPose pose;
        pose.time_ns = cur_ns;
        pose.R = rot;
        pose.p = pos;
        pose.v = vel;
        pose.gyro = gyr;
        pose.acc = acc_w;
        poses.push_back(pose);

        F.setIdentity();
        cov_w.setZero();
        F.block<3, 3>(0, 0) = ExpMat(Vec3(-gyr * dt));
        F.block<3, 3>(0, 9) = -Mat3::Identity() * dt;
        F.block<3, 3>(3, 6) = Mat3::Identity() * dt;
        F.block<3, 3>(6, 0) = -rot * Hat(acc) * dt;
        F.block<3, 3>(6, 12) = -rot * dt;
        cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
        cov_w.block<3, 3>(6, 6) = rot * cov_acc.asDiagonal() * rot.transpose() * dt * dt;
        cov_w.block<3, 3>(9, 9).diagonal() = cov_bg * dt * dt;
        cov_w.block<3, 3>(12, 12).diagonal() = cov_ba * dt * dt;
        x.cov = F * x.cov * F.transpose() + cov_w;
        phi_ = F * phi_;

        pos += vel * dt + 0.5 * acc_w * dt * dt;
        vel += acc_w * dt;
        rot = rot * ExpMat(Vec3(gyr * dt));
    }
    if (poses.empty()) {
        ImuPose pose;
        pose.time_ns = last_end_ns;
        pose.R = rot;
        pose.p = pos;
        pose.v = vel;
        pose.gyro = gyr;
        pose.acc = acc_w;
        poses.push_back(pose);
    }

    // from the last sample to the end of the frame
    const int64_t imu_end_ns = std::max(imus.back()->time_ns, last_end_ns);
    const double dt_end = NS_TO_S(end_ns - imu_end_ns);
    x.v = vel + acc_w * dt_end;
    x.R = rot * ExpMat(Vec3(gyr * dt_end));
    x.p = pos + vel * dt_end + 0.5 * acc_w * dt_end * dt_end;
    if (!measures.imu_.empty()) last_imu_ = measures.imu_.back();
    last_end_ns = end_ns;

    // undistort, walking back from the end of the frame
    const Mat3 R_il = param->extrin_il_.q_.toRotationMatrix();
    const Vec3 t_il = param->extrin_il_.t_;
    const PointCloud& cloud = *measures.lidar_;
    scan->resize(cloud.size());
    int idx = static_cast<int>(cloud.size()) - 1;
    for (int i = static_cast<int>(poses.size()) - 1; i >= 0 && idx >= 0; --i) {
        const ImuPose& head = poses[i];
        // what is left before the first pose can only be a point right on the start of the frame
        for (; idx >= 0 && (i == 0 || cloud.points[idx].GetTimeNs() >= head.time_ns); --idx) {
            const Point& pt = cloud.points[idx];
            const double dt = NS_TO_S(pt.GetTimeNs() - head.time_ns);
            const Mat3 R_i = head.R * ExpMat(Vec3(head.gyro * dt));
            const Vec3 T_ei = head.p + head.v * dt + 0.5 * head.acc * dt * dt - x.p;
            const Vec3 p_l = pt.getVector3fMap().cast<double>();
            (*scan)[idx] = R_il.transpose() * (x.R.transpose() * (R_i * (R_il * p_l + t_il) + T_ei) - t_il);
        }
    }
}

std::vector<size_t> FastLio::UniformSampleIndices(const std::vector<Vec3>& scan, double size) const {
    struct Best {
        size_t idx;
        double d2;
    };
    std::unordered_map<VOXEL_LOCATION, size_t> slot;  // voxel -> index in `best`
    slot.reserve(scan.size());
    std::vector<Best> best;
    for (size_t i = 0; i < scan.size(); ++i) {
        const Vec3& p = scan[i];
        const VOXEL_LOCATION loc(p, size);
        const Vec3 center = (Vec3(loc.x, loc.y, loc.z) + Vec3::Constant(0.5)) * size;
        const double d2 = (p - center).squaredNorm();
        const auto ins = slot.emplace(loc, best.size());
        if (ins.second) {
            best.push_back({i, d2});
        } else if (d2 < best[ins.first->second].d2) {
            best[ins.first->second] = {i, d2};
        }
    }
    std::vector<size_t> out;
    out.reserve(best.size());
    for (const Best& b : best) out.push_back(b.idx);
    return out;
}

std::vector<Vec3> FastLio::UniformSample(const std::vector<Vec3>& scan, double size) const {
    const std::vector<size_t> idx = UniformSampleIndices(scan, size);
    std::vector<Vec3> out;
    out.reserve(idx.size());
    for (const size_t i : idx) out.push_back(scan[i]);
    return out;
}

void FastLio::SetPoints(const std::vector<Vec3>& scan) {
    const Mat3 R_il = param->extrin_il_.q_.toRotationMatrix();
    const Vec3 t_il = param->extrin_il_.t_;
    points.clear();
    points.resize(scan.size());
    for (size_t i = 0; i < scan.size(); ++i) {
        FastLioPoint& pt = points[i];
        pt.body = R_il * scan[i] + t_il;
        pt.var = R_il * BodyPointCov(scan[i], param->lba_.range_err, param->lba_.angle_err) * R_il.transpose();
    }
}

int FastLio::Match() {
    const Mat3 R = x.R;
    const Vec3 p = x.p;
    const Mat3 rot_var = x.cov.block<3, 3>(0, 0);
    const Mat3 tsl_var = x.cov.block<3, 3>(3, 3);
    const double r_base = param->fast_lio_.r_base;
    const int num_pt = static_cast<int>(points.size());
#pragma omp parallel for num_threads(param->num_threads) schedule(dynamic) if (num_pt > kMinParallelPoints)
    for (int i = 0; i < num_pt; ++i) {
        FastLioPoint& pt = points[i];
        const Mat3 phat = Hat(pt.body);
        const Mat3 var_world = R * pt.var * R.transpose() + phat * rot_var * phat.transpose() + tsl_var;
        pt.world = R * pt.body + p;
        PlaneHit hit;
        pt.valid = lba->Query(LbaPoint{pt.world, var_world}, &hit);
        if (!pt.valid) continue;
        pt.normal = hit.normal;
        pt.residual = hit.signed_dist;
        pt.r_inv = 1.0 / (r_base + hit.plane_sigma + hit.normal.dot(var_world * hit.normal));
    }
    int matched = 0;
    for (const FastLioPoint& pt : points) matched += pt.valid;
    return matched;
}

bool FastLio::IteratedUpdate() {
    using MatN = FastLioState::MatN;
    using VecN = FastLioState::VecN;
    constexpr int DIM = FastLioState::DIM;
    const FastLioState x_prop = x;
    const MatN cov_inv = x.cov.inverse();
    MatN G = MatN::Zero(), H_T_H = MatN::Zero();
    Mat3 nnt = Mat3::Zero();
    int rematch_num = 0;
    int matched = 0;
    for (int iter = 0; iter < kMaxIter; ++iter) {
        matched = Match();

        Eigen::Matrix<double, 6, 6> HTH = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> HTz = Eigen::Matrix<double, 6, 1>::Zero();
        nnt.setZero();
        const Mat3 Rt = x.R.transpose();
        for (const FastLioPoint& pt : points) {
            if (!pt.valid) continue;
            Eigen::Matrix<double, 6, 1> jac;
            jac.head<3>() = Hat(pt.body) * Rt * pt.normal;
            jac.tail<3>() = pt.normal;
            HTH += pt.r_inv * jac * jac.transpose();
            HTz -= pt.r_inv * jac * pt.residual;
            nnt += pt.normal * pt.normal.transpose();
        }

        H_T_H.block<6, 6>(0, 0) = HTH;
        const MatN K_1 = (H_T_H + cov_inv).inverse();
        G.block<DIM, 6>(0, 0) = K_1.block<DIM, 6>(0, 0) * HTH;
        const VecN vec = x_prop - x;
        const VecN solution = K_1.block<DIM, 6>(0, 0) * HTz + vec - G.block<DIM, 6>(0, 0) * vec.head<6>();
        x += solution;

        const bool converged =
            solution.segment<3>(0).norm() * 57.3 < 0.01 && solution.segment<3>(3).norm() * 100 < 0.015;
        if (converged || (rematch_num == 0 && iter == kMaxIter - 2)) ++rematch_num;
        if (rematch_num >= 2 || iter == kMaxIter - 1) {
            x.cov = (MatN::Identity() - G) * x.cov;
            break;
        }
    }
    const Eigen::SelfAdjointEigenSolver<Mat3> saes(nnt);
    LOG(INFO) << "Points: " << points.size() << " matched " << matched << " min eigen " << saes.eigenvalues()[0];
    return saes.eigenvalues()[0] >= kMinNormalEigen;
}

int FastLio::SubFrameCount(const MeasureGroup& measures) const {
    const FastLioParam& fp = param->fast_lio_;
    if (fp.sub_max <= 1 || measures.imu_.size() < 2) return 1;
    if (fp.sub_std_acc <= 0.0 && fp.sub_std_gyr <= 0.0) return fp.sub_max;
    Vec3 acc_sum = Vec3::Zero(), acc_sq = Vec3::Zero(), gyr_sum = Vec3::Zero(), gyr_sq = Vec3::Zero();
    for (const Imu& imu : measures.imu_) {
        acc_sum += imu.linear_acceleration;
        acc_sq += imu.linear_acceleration.cwiseProduct(imu.linear_acceleration);
        gyr_sum += imu.angular_velocity;
        gyr_sq += imu.angular_velocity.cwiseProduct(imu.angular_velocity);
    }
    const double m = static_cast<double>(measures.imu_.size());
    const auto std_of = [m](const Vec3& sum, const Vec3& sq) {
        const Vec3 mean = sum / m;
        return std::sqrt(std::max(0.0, (sq / m - mean.cwiseProduct(mean)).sum() / 3.0));
    };
    const double sa = std_of(acc_sum, acc_sq);
    const double sg = std_of(gyr_sum, gyr_sq);
    const double ratio = std::max(fp.sub_std_acc > 0.0 ? sa / fp.sub_std_acc : 0.0,
                                  fp.sub_std_gyr > 0.0 ? sg / fp.sub_std_gyr : 0.0);
    return std::min(fp.sub_max, std::max(1, static_cast<int>(std::ceil(fp.sub_max * ratio))));
}

void FastLio::ProcessSubFrames(MeasureGroup& measures, int n) {
    using MatN = FastLioState::MatN;
    using VecN = FastLioState::VecN;
    struct Node {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        FastLioState prior;  // after the propagation to the end of the slice, cov = P-
        FastLioState post;   // after its update
        MatN phi;            // error-state transition from the previous slice's end
        Eigen::aligned_vector<FastLioPoint> pts;  // body frame at the end of the slice
    };
    Eigen::aligned_vector<Node> nodes(n);
    const FastLioParam& fast_opt = param->fast_lio_;
    const int64_t t0 = last_end_ns;
    const int64_t t1 = measures.mea_end_;
    const PointCloud& cloud = *measures.lidar_;
    size_t pi = 0, ii = 0;
    MeasureGroup slice;
    for (int j = 0; j < n; ++j) {
        const bool last = j == n - 1;
        const int64_t tj = last ? t1 : t0 + (t1 - t0) * (j + 1) / n;
        slice.imu_.clear();
        slice.lidar_->clear();
        slice.mea_end_ = tj;
        while (ii < measures.imu_.size() && (last || measures.imu_[ii].time_ns <= tj))
            slice.imu_.push_back(measures.imu_[ii++]);
        while (pi < cloud.size() && (last || cloud.points[pi].GetTimeNs() <= tj))
            slice.lidar_->push_back(cloud.points[pi++]);

        std::vector<Vec3> scan;
        Propagate(slice, tj, &scan);
        nodes[j].phi = phi_;
        nodes[j].prior = x;
        std::vector<Vec3> down = UniformSample(scan, fast_opt.down_size);
        if (static_cast<int>(down.size()) < kMinScanPoints / n) down = UniformSample(scan, fast_opt.down_size / 2);
        SetPoints(down);
        if (last && !tdcp_epoch_.empty()) PrepareTdcp();
        if (!points.empty()) {
            if (IteratedUpdate()) {
                if (degrade_cnt_ > 0) degrade_cnt_--;
            } else {
                degrade_cnt_++;
            }
        }
        nodes[j].post = x;
        nodes[j].pts = points;
    }

    // backward smoother (FR-LIO Eq. 12-14): the correction the last update made is pushed back through the slices
    Eigen::aligned_vector<FastLioState> smooth(n);
    smooth[n - 1] = nodes[n - 1].post;
    MatN P_next = nodes[n - 1].post.cov;
    for (int j = n - 2; j >= 0; --j) {
        const MatN& P_post = nodes[j].post.cov;
        const MatN& P_prior_next = nodes[j + 1].prior.cov;
        const MatN G = P_prior_next.ldlt().solve(nodes[j + 1].phi * P_post).transpose();  // P+ Phi^T (P-)^-1
        const VecN dx = smooth[j + 1] - nodes[j + 1].prior;
        smooth[j] = nodes[j].post;
        smooth[j] += G * dx;
        smooth[j].cov = P_post + G * (P_next - P_prior_next) * G.transpose();
        P_next = smooth[j].cov;
    }

    // the whole frame at the frame end, with the smoothed pose of every slice
    x = smooth[n - 1];
    const Mat3 Rt = x.R.transpose();
    Eigen::aligned_vector<FastLioPoint> all;
    std::vector<Vec3> pos;
    for (int j = 0; j < n; ++j) {
        const Mat3 Rr = Rt * smooth[j].R;
        const Vec3 tr = Rt * (smooth[j].p - x.p);
        for (const FastLioPoint& pt : nodes[j].pts) {
            FastLioPoint q = pt;
            if (j < n - 1) {
                q.body = Rr * pt.body + tr;
                q.var = Rr * pt.var * Rr.transpose();
            }
            pos.push_back(q.body);
            all.push_back(q);
        }
    }

    points.clear();
    for (const size_t i : UniformSampleIndices(pos, fast_opt.down_size)) points.push_back(all[i]);
}

void FastLio::ProcessMeasurement(MeasureGroup& measures) {
    Timer timer;
    for (const Imu& imu : measures.imu_) lba->AddImu(imu.time_ns, imu.angular_velocity, imu.linear_acceleration);

    // LIGO: the frame ends on a GNSS epoch; its TDCP rows join the update at the frame end
    sol_t sol{};
    bool gnss_epoch_ok = false;
    tdcp_epoch_.clear();
    tdcp_.clear();
    if (param->fast_lio_.mode == FastLioMode::LIGO && measures.obs_id >= 0 && obss.n > 0) {
        CHECK_EQ(measures.obs_time_ns, measures.mea_end_);
        gnss_epoch_ok = ProcessGnssEpoch(obss, nav, *param, sat_tracks_, measures.obs_id, measures.obs_time_ns,
                                         rtk_state.aligned && param->tdcp_enable, sol, &tdcp_epoch_);
    }

    const int n_sub = SubFrameCount(measures);
    if (n_sub <= 1) {
        std::vector<Vec3> scan;
        Propagate(measures, measures.mea_end_, &scan);

        const double down_size = param->fast_lio_.down_size;
        std::vector<Vec3> down = UniformSample(scan, down_size);
        if (static_cast<int>(down.size()) < kMinScanPoints) down = UniformSample(scan, down_size / 2);
        SetPoints(down);
        if (!tdcp_epoch_.empty()) PrepareTdcp();

        if (!points.empty()) {
            if (IteratedUpdate()) {
                if (degrade_cnt_ > 0) degrade_cnt_--;
            } else {
                degrade_cnt_++;
            }
        }
    } else {
        ProcessSubFrames(measures, n_sub);
    }
    const double t_update = timer.ElapsedMiniSeconds(true);

    LbaState odom;
    odom.R = x.R;
    odom.p = x.p;
    odom.v = x.v;
    odom.bg = x.bg;
    odom.ba = x.ba;
    std::vector<LbaPoint> frame_points;
    frame_points.reserve(points.size());
    for (const FastLioPoint& pt : points) frame_points.push_back(LbaPoint{pt.body, pt.var});
    if (lba->AddFrame(measures.mea_end_, odom, frame_points, gravity)) {
        x.R = lba->Newest().R;
        x.p = lba->Newest().p;
    }
    for (FastLioPoint& pt : points) pt.world = x.R * pt.body + x.p;
    LOG(INFO) << "time [ms]: update " << t_update << " lba " << timer.ElapsedMiniSeconds() << " raw points "
              << measures.lidar_->size() << " degrade " << degrade_cnt_ << " sub " << n_sub << " bg " << x.bg.transpose() << " ba "
              << x.ba.transpose();
}

double FastLio::AntennaRange(const Mat3& R, const Vec3& p, const Vec3& sv_pos, bool nav_cols,
                             Eigen::Matrix<double, 1, FastLioState::DIM + RTKState::DIM>* H) const {
    const Vec3& lever = param->level_arm;
    const Vec3 p_ant = R * lever + p;
    const Mat3 rot = rtk_state.RotEcefEnu();
    const Vec3 p_ecef = rot * p_ant + rtk_state.t_ecef_enu;
    double e[3];
    const double range = geodist(sv_pos.data(), p_ecef.data(), e);
    if (!H) return range;

    // d(range)/d(receiver ECEF), Sagnac term included
    const Eigen::RowVector3d d_range = -Eigen::Map<const Vec3>(e).transpose() +
                                       Eigen::RowVector3d(-OMGE / CLIGHT * sv_pos.y(), OMGE / CLIGHT * sv_pos.x(), 0.0);
    const Eigen::RowVector3d d_range_local = d_range * rot;
    H->setZero();
    if (nav_cols) {
        // R Exp(d theta) lever = R lever - R [lever]x d theta
        H->block<1, 3>(0, 0) = -d_range_local * R * Hat(lever);
        H->block<1, 3>(0, 3) = d_range_local;
    }
    // R_init Rz(yaw) v, d/d(yaw) = R (ez x v)
    (*H)(0, FastLioState::DIM + RTKState::YAW) = (d_range_local * Vec3::UnitZ().cross(p_ant)).value();
    H->block<1, 3>(0, FastLioState::DIM + RTKState::TRN) = d_range;
    return range;
}

double FastLio::TdcpResidual(const TdcpObs& obs,
                             Eigen::Matrix<double, 1, FastLioState::DIM + RTKState::DIM>* H) const {
    // lambda*dPhi + c*d(dts) = (range_{k+1} - range_k) + (clk_{k+1} - clk_k): the pose and the clocks of epoch k
    // are held at the anchor, only the extrinsic acts on both ranges
    Mat3 R_anchor;
    Vec3 p_anchor;
    Eigen::Matrix<double, 1, FastLioState::DIM + RTKState::DIM> H_anchor;
    const double range = AntennaRange(x.R, x.p, obs.sv_pos, true, H);
    const double range_anchor = AntennaRange(R_anchor, p_anchor, obs.sv_pos_anchor, false, H ? &H_anchor : nullptr);
    const int s = obs.sys_idx;
    const double pred = range - range_anchor + rtk_state.clk(s) - rtk_state.clk_anchor(s);
    if (H) {
        *H -= H_anchor;
        (*H)(0, FastLioState::DIM + RTKState::CLK + s) = 1.0;
    }
    return obs.cp_diff + CLIGHT * (obs.sv_dt - obs.sv_dt_anchor) - pred;
}

void FastLio::PrepareTdcp() {
    // Residual without the receiver clock step.  The step is common to every system (the inter-system biases
    // cancel between epochs), so the median across satellites estimates it.
    constexpr int N = FastLioState::DIM;
    constexpr int NG = RTKState::DIM;
    using RowJ = Eigen::Matrix<double, 1, N + NG>;
    Eigen::Matrix<double, N + NG, N + NG> cov;
    cov.topLeftCorner<N, N>() = x.cov;
    cov.topRightCorner<N, NG>() = rtk_state.P_ng;
    cov.bottomLeftCorner<NG, N>() = rtk_state.P_ng.transpose();
    cov.bottomRightCorner<NG, NG>() = rtk_state.P;

    std::vector<TdcpObs> rows;
    std::vector<double> res, var_innov;
    Mat3 R_anchor;
    Vec3 p_anchor;
    for (TdcpObs& obs : tdcp_epoch_) {
        // the phase noise alone ignores multipath, the anchor pose error and the antenna-to-IMU mismatch:
        // tdcp_std at zenith for each of the two epochs, scaled by 1/sin(el), as the offline graph weighs them
        if (param->tdcp_std > 0.0) obs.var = 2.0 * (param->tdcp_std / obs.sin_el) * (param->tdcp_std / obs.sin_el);
        RowJ H_cur, H_anchor;
        const double range = AntennaRange(x.R, x.p, obs.sv_pos, true, &H_cur);
        const double range_anchor = AntennaRange(R_anchor, p_anchor, obs.sv_pos_anchor, false, &H_anchor);
        rows.push_back(obs);
        res.push_back(obs.cp_diff + CLIGHT * (obs.sv_dt - obs.sv_dt_anchor) - (range - range_anchor));
        // innovation variance without the clock column: the median below takes the clock step out
        const RowJ h = H_cur - H_anchor;
        var_innov.push_back((h * cov * h.transpose())(0) + obs.var);
    }
    const int num = static_cast<int>(rows.size());
    tdcp_.clear();
    if (num == 0) return;
    std::vector<double> sorted = res;
    std::nth_element(sorted.begin(), sorted.begin() + num / 2, sorted.end());
    const double clk_step = sorted[num / 2];

    // hold the clocks of epoch k as the anchor and predict epoch k+1 from the median step
    rtk_state.clk_anchor = rtk_state.clk;
    rtk_state.clk.array() += clk_step;

    // LIVO-prior chi-square test (FAST-LIVGO eq. 14) on each clock-free innovation
    double sq_sum = 0.0;
    for (int i = 0; i < num; ++i) {
        const double r = res[i] - clk_step;
        if (r * r / var_innov[i] < param->tdcp_chi2 && std::abs(r) < param->tdcp_outlier) {
            tdcp_.push_back(rows[i]);
            sq_sum += r * r;
        }
    }
    const int num_valid = static_cast<int>(tdcp_.size());
    LOG(INFO) << "TDCP: " << num_valid << "/" << num << " valid, rms " << std::sqrt(sq_sum / std::max(num_valid, 1))
              << " m, clock step " << clk_step << " m, yaw_ecef_enu " << rtk_state.yaw;
    if (num_valid < param->tdcp_min_sat) tdcp_.clear();
}

void FastLio::InitGnss(const sol_t& sol) {
    // pntpos reports the GPS clock and the GLO/GAL/BDS offsets from it
    rtk_state.clk(0) = CLIGHT * sol.dtr[0];
    rtk_state.clk(1) = CLIGHT * (sol.dtr[0] + sol.dtr[1]);
    rtk_state.clk(2) = CLIGHT * (sol.dtr[0] + sol.dtr[2]);
    rtk_state.clk(3) = CLIGHT * (sol.dtr[0] + sol.dtr[3]);
    rtk_state.clk_anchor = rtk_state.clk;
    rtk_state.yaw = 0.0;

    rtk_state.P.setZero();
    rtk_state.P.block<4, 4>(RTKState::CLK, RTKState::CLK) =
        param->std_clk_init * param->std_clk_init * Eigen::Matrix4d::Identity();
    rtk_state.P(RTKState::YAW, RTKState::YAW) = param->std_ext_yaw_init * param->std_ext_yaw_init;
    rtk_state.P.block<3, 3>(RTKState::TRN, RTKState::TRN) =
        param->std_ext_pos_init * param->std_ext_pos_init * Eigen::Matrix3d::Identity();
    rtk_state.P_ng.setZero();
    rtk_state.aligned = true;
}
}  // namespace kuma_lio
