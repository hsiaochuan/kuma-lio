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
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "timer.h"

namespace kuma_lio {
namespace {
constexpr int kMinParallelPoints = 2048;
constexpr int kMaxIter = 4;
constexpr int kMinScanPoints = 500;     // below this the frame is downsampled at half the size
constexpr double kMinNormalEigen = 14;  // smallest eigenvalue of sum(n n^T) over a frame's matches
// error-state blocks of FastLioState; a LiDAR row touches only the pose, ROT and POS, the first POSE entries
constexpr int ROT = FastLioState::ROT, POS = FastLioState::POS, VEL = FastLioState::VEL, BG = FastLioState::BG,
              BA = FastLioState::BA, YAW = FastLioState::YAW, TEC = FastLioState::TEC, CLK = FastLioState::CLK;
constexpr int POSE = 6;
static_assert(ROT == 0 && POS == 3, "the pose block must be the first POSE entries");
}  // namespace

void FastLio::Init(const std::string& pgo_config) {
    param->lba_.num_threads = param->num_threads;
    lba = std::make_unique<LocalBA>(param->lba_);
    LOG(INFO) << "local BA map shared with the odometry: voxel " << param->lba_.voxel_size << " window "
              << param->lba_.win_size;
    if (!pgo_config.empty()) {
        pgo::PgoParam pgo_prm;
        pgo::LoadPgoParam(pgo_config, &pgo_prm);
        pgo_prm.keep_kf_points = true;
        const double fixed_var = param->lba_.range_err * param->lba_.range_err + param->lba_.pose_var;
        loop = std::make_unique<LoopClosure>(pgo_prm, lba.get(), fixed_var);
        LOG(INFO) << "loop closure parameters from " << pgo_config;
    }
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
    Eigen::Matrix4d q_clk = Eigen::Matrix4d::Zero();
    if (gnss_aligned_)
        q_clk = param->std_clk * param->std_clk * Eigen::Matrix4d::Ones() +
                param->std_isb * param->std_isb * Eigen::Matrix4d::Identity();
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
        F.block<3, 3>(ROT, ROT) = ExpMat(Vec3(-gyr * dt));
        F.block<3, 3>(ROT, BG) = -Mat3::Identity() * dt;
        F.block<3, 3>(POS, VEL) = Mat3::Identity() * dt;
        F.block<3, 3>(VEL, ROT) = -rot * Hat(acc) * dt;
        F.block<3, 3>(VEL, BA) = -rot * dt;
        cov_w.block<3, 3>(ROT, ROT).diagonal() = cov_gyr * dt * dt;
        cov_w.block<3, 3>(VEL, VEL) = rot * cov_acc.asDiagonal() * rot.transpose() * dt * dt;
        cov_w.block<3, 3>(BG, BG).diagonal() = cov_bg * dt * dt;
        cov_w.block<3, 3>(BA, BA).diagonal() = cov_ba * dt * dt;
        cov_w.block<4, 4>(CLK, CLK) = q_clk * dt;
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
    const Mat3 rot_var = x.cov.block<3, 3>(ROT, ROT);
    const Mat3 tsl_var = x.cov.block<3, 3>(POS, POS);
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
    const FastLioState x_prop = x;
    const MatN cov_inv = x.cov.inverse();
    MatN G = MatN::Zero(), H_T_H = MatN::Zero();
    Mat3 nnt = Mat3::Zero();
    int rematch_num = 0;
    int matched = 0;
    for (int iter = 0; iter < kMaxIter; ++iter) {
        matched = Match();

        Eigen::Matrix<double, POSE, POSE> HTH = Eigen::Matrix<double, POSE, POSE>::Zero();
        Eigen::Matrix<double, POSE, 1> HTz = Eigen::Matrix<double, POSE, 1>::Zero();
        nnt.setZero();
        const Mat3 Rt = x.R.transpose();
        for (const FastLioPoint& pt : points) {
            if (!pt.valid) continue;
            Eigen::Matrix<double, POSE, 1> jac;
            jac.segment<3>(ROT) = Hat(pt.body) * Rt * pt.normal;
            jac.segment<3>(POS) = pt.normal;
            HTH += pt.r_inv * jac * jac.transpose();
            HTz -= pt.r_inv * jac * pt.residual;
            nnt += pt.normal * pt.normal.transpose();
        }

        // the LiDAR rows touch only the pose; the TDCP rows also the yaw, t_ecef_enu and the clocks
        H_T_H.setZero();
        H_T_H.block<POSE, POSE>(ROT, ROT) = HTH;
        VecN HTz_all = VecN::Zero();
        HTz_all.segment<POSE>(ROT) = HTz;
        if (tdcp_.valid) AddTdcpObservations(iter == 0, x_prop, &H_T_H, &HTz_all);

        const MatN K_1 = (H_T_H + cov_inv).inverse();
        G = K_1 * H_T_H;
        const VecN vec = x_prop - x;
        const VecN solution = K_1 * HTz_all + vec - G * vec;
        x += solution;

        const bool converged =
            solution.segment<3>(ROT).norm() * 57.3 < 0.01 && solution.segment<3>(POS).norm() * 100 < 0.015;
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
    const double ratio =
        std::max(fp.sub_std_acc > 0.0 ? sa / fp.sub_std_acc : 0.0, fp.sub_std_gyr > 0.0 ? sg / fp.sub_std_gyr : 0.0);
    return std::min(fp.sub_max, std::max(1, static_cast<int>(std::ceil(fp.sub_max * ratio))));
}

void FastLio::ProcessSubFrames(MeasureGroup& measures, int n) {
    using MatN = FastLioState::MatN;
    using VecN = FastLioState::VecN;
    struct Node {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        FastLioState prior;                       // after the propagation to the end of the slice, cov = P-
        FastLioState post;                        // after its update
        MatN phi;                                 // error-state transition from the previous slice's end
        Eigen::aligned_vector<FastLioPoint> pts;  // body frame at the end of the slice
    };
    Eigen::aligned_vector<Node> nodes(n);
    const FastLioParam& fast_opt = param->fast_lio_;
    const int64_t t0 = last_end_ns;
    const int64_t t1 = measures.mea_end_;
    const PointCloud& cloud = *measures.lidar_;
    size_t pi = 0, ii = 0;
    MeasureGroup slice;
    const bool tdcp_valid = tdcp_.valid;
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
        tdcp_.valid = tdcp_valid && last;  // the epoch is at the frame end: its rows are the last slice's
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

    // LIGO: the frame ends on a GNSS epoch; its TDCP rows join the update at the frame end, and until the alignment
    // succeeds the epoch feeds it
    tdcp_ = Tdcp();
    std::optional<GnssEpoch> gnss_epoch;
    if (param->fast_lio_.mode == FastLioMode::LIGO && measures.obs_id >= 0 && obss.n > 0) {
        CHECK_EQ(measures.obs_time_ns, measures.mea_end_);
        // SPP and TDCP rows of the epoch
        sol_t sol{};
        GnssEpoch epoch;
        if (ProcessGnssEpoch(obss, nav, *param, sat_tracks_, measures.obs_id, measures.obs_time_ns, param->tdcp_enable,
                             sol, &epoch.tdcp_obs)) {
            epoch.time_ns = measures.obs_time_ns;
            epoch.ecef_spp = Vec3(sol.rr[0], sol.rr[1], sol.rr[2]);
            epoch.clk_spp =
                CLIGHT * Vec4(sol.dtr[0], sol.dtr[0] + sol.dtr[1], sol.dtr[0] + sol.dtr[2], sol.dtr[0] + sol.dtr[3]);
            // once aligned, the rows anchored at the previous frame end go to the update with the anchor
            if (param->tdcp_enable && gnss_aligned_) {
                tdcp_.ant_anchor = x.R * param->level_arm + x.p;  // x is still the previous frame end
                tdcp_.clk_anchor = x.clk;
                for (const TdcpObs& obs : epoch.tdcp_obs)
                    if (obs.anchor_time_ns == last_end_ns) tdcp_.obs.push_back(obs);
                tdcp_.valid = static_cast<int>(tdcp_.obs.size()) >= param->tdcp_min_sat;
            }
            gnss_epoch = std::move(epoch);
        }
    }

    const int n_sub = SubFrameCount(measures);
    if (n_sub <= 1) {
        std::vector<Vec3> scan;
        Propagate(measures, measures.mea_end_, &scan);

        const double down_size = param->fast_lio_.down_size;
        std::vector<Vec3> down = UniformSample(scan, down_size);
        if (static_cast<int>(down.size()) < kMinScanPoints) down = UniformSample(scan, down_size / 2);
        SetPoints(down);

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
    if (tdcp_.num_used > 0) {
        // post-fit at the updated state, before the BA moves it
        std::vector<double> residuals;
        TdcpResiduals(x, &residuals, nullptr);
        double ss = 0.0, nn = 0.0;
        for (size_t k = 0; k < residuals.size(); ++k) {
            if (!tdcp_.mask[k]) continue;
            const double s_el = param->tdcp_std / tdcp_.obs[k].sin_el;
            ss += residuals[k] * residuals[k];
            nn += residuals[k] * residuals[k] / (2.0 * s_el * s_el);
        }
        tdcp_.postfit_rms = std::sqrt(ss / tdcp_.num_used);
        tdcp_.postfit_nrms = std::sqrt(nn / tdcp_.num_used);
    }

    LbaState odom;
    odom.R = x.R;
    odom.p = x.p;
    odom.v = x.v;
    odom.bg = x.bg;
    odom.ba = x.ba;
    std::vector<LbaPoint> frame_points;
    frame_points.reserve(points.size());
    for (const FastLioPoint& pt : points) frame_points.push_back(LbaPoint{pt.body, pt.var});
    // Voxel-SLAM keyframe_loading, before the frame goes in so one Refresh covers both
    if (loop) loop->ReloadKeyframe(x.p);
    if (lba->AddFrame(measures.mea_end_, odom, std::move(frame_points), gravity)) {
        LocalBA::MargScan scan;
        Eigen::Isometry3d dx;
        if (loop && lba->TakeMarginalized(&scan) &&
            loop->AddScan(scan, global_offset_time + NS_TO_S(scan.time_ns), &dx)) {
            x.R = dx.linear() * x.R;
            x.p = dx.linear() * x.p + dx.translation();
            x.v = dx.linear() * x.v;
        }
        x.R = lba->Newest().R;
        x.p = lba->Newest().p;
    }
    if (gnss_epoch && !gnss_aligned_) AlignGnss(*gnss_epoch);
    for (FastLioPoint& pt : points) pt.world = x.R * pt.body + x.p;
    LOG(INFO) << "time [ms]: update " << t_update << " lba " << timer.ElapsedMiniSeconds() << " raw points "
              << measures.lidar_->size() << " degrade " << degrade_cnt_ << " sub " << n_sub << " bg "
              << x.bg.transpose() << " ba " << x.ba.transpose();
    if (tdcp_.num_used + tdcp_.num_gated > 0) {
        LOG(INFO) << "TDCP: " << tdcp_.num_used << "/" << tdcp_.obs.size() << " rows, postfit rms " << tdcp_.postfit_rms
                  << " m, normalized " << tdcp_.postfit_nrms << ", yaw " << x.yaw * R2D << " deg, clk step "
                  << x.clk(0) - tdcp_.clk_anchor(0) << " m";
    }
}

void FastLio::TdcpResiduals(const FastLioState& x, std::vector<double>* residuals,
                            Eigen::aligned_vector<FastLioState::VecN>* jacos) const {
    // lambda (phi_k+1 - phi_k) + c (dt_sv_k+1 - dt_sv_k) = rho_k+1 - rho_k + clk_k+1 - clk_k
    const Vec3& l = param->level_arm;
    const Mat3 Rz = Eigen::AngleAxisd(x.yaw, Vec3::UnitZ()).toRotationMatrix();
    const Mat3 R_el = R_ecef_enu_ * Rz;
    const Mat3 dR_el = R_ecef_enu_ * Hat(Vec3::UnitZ()) * Rz;  // d R_el / d yaw
    const Vec3 a1 = x.R * l + x.p;
    const Vec3& a0 = tdcp_.ant_anchor;
    const Vec3 r1 = R_el * a1 + x.t_ecef_enu;
    const Vec3 r0 = R_el * a0 + x.t_ecef_enu;
    const Mat3 J_rot = R_el * x.R * Hat(l);  // d r1 = -R_el R [l]x dtheta for R <- R Exp(dtheta)
    residuals->clear();
    if (jacos) jacos->clear();
    for (const TdcpObs& obs : tdcp_.obs) {
        double e1[3], e0[3];  // unit vectors from the receiver to the satellite
        const double rho1 = geodist(obs.sv_pos.data(), r1.data(), e1);
        const double rho0 = geodist(obs.sv_pos_anchor.data(), r0.data(), e0);
        const Vec3 u1(e1[0], e1[1], e1[2]), u0(e0[0], e0[1], e0[2]);
        const double z = obs.cp_diff + CLIGHT * (obs.sv_dt - obs.sv_dt_anchor);
        residuals->push_back(rho1 - rho0 + x.clk(obs.sys_idx) - tdcp_.clk_anchor(obs.sys_idx) - z);
        if (!jacos) continue;
        FastLioState::VecN J = FastLioState::VecN::Zero();
        J.segment<3>(ROT) = J_rot.transpose() * u1;
        J.segment<3>(POS) = -R_el.transpose() * u1;
        J(YAW) = -u1.dot(dR_el * a1) + u0.dot(dR_el * a0);
        J.segment<3>(TEC) = u0 - u1;  // t_ecef_enu cancels up to the change of the line of sight
        J(CLK + obs.sys_idx) = 1.0;
        jacos->push_back(J);
    }
}

void FastLio::AddTdcpObservations(bool if_gate, const FastLioState& x_prop, FastLioState::MatN* HTH,
                                  FastLioState::VecN* HTz) {
    std::vector<double> residuals;
    Eigen::aligned_vector<FastLioState::VecN> jacobians;
    TdcpResiduals(x, &residuals, &jacobians);
    const size_t n = residuals.size();
    // one row: tdcp_std / sin(el) at zenith for each of its two epochs
    std::vector<double> var(n);
    for (size_t k = 0; k < n; ++k) {
        const double s_el = param->tdcp_std / tdcp_.obs[k].sin_el;
        var[k] = 2.0 * s_el * s_el;
    }
    if (if_gate) {
        tdcp_.mask.assign(n, true);
        tdcp_.num_used = 0;
        for (size_t k = 0; k < n; ++k) {
            const double s = (jacobians[k].transpose() * x_prop.cov * jacobians[k])(0) + var[k];
            if (residuals[k] * residuals[k] / s > param->tdcp_chi2) tdcp_.mask[k] = false;
            else
                ++tdcp_.num_used;
        }
        tdcp_.num_gated = static_cast<int>(n) - tdcp_.num_used;
        if (tdcp_.num_used < param->tdcp_min_sat) {
            tdcp_.num_gated = static_cast<int>(n);
            tdcp_.num_used = 0;
            tdcp_.valid = false;
        }
        if (!tdcp_.valid) return;
    }
    for (size_t k = 0; k < n; ++k) {
        if (!tdcp_.mask[k]) continue;
        const double w = 1.0 / var[k];
        *HTH += w * jacobians[k] * jacobians[k].transpose();
        *HTz -= w * jacobians[k] * residuals[k];
    }
}

void FastLio::AlignGnss(const GnssEpoch& epoch) {
    // step 1: the first SPP fix gives R_ecef_enu, the ENU frame there; only the yaw about its z is unknown
    const Vec3 ant_local = x.R * param->level_arm + x.p;
    if (align_ecefs_.empty()) R_ecef_enu_ = EnuRotation(epoch.ecef_spp);
    if (align_locals_.empty() || (align_locals_.back() - ant_local).norm() > 0.3) {
        align_locals_.push_back(ant_local);
        align_ecefs_.push_back(epoch.ecef_spp);
    }
    if (param->ali_refine) align_window_[epoch.time_ns] = {ant_local, epoch};
    // step 2: once the horizontal track is long enough, GnssAlignment gives the yaw and t_ecef_enu
    double yaw = 0.0;
    Vec3 t;
    if (!GnssAlignment(R_ecef_enu_, align_locals_, align_ecefs_, param->ali_min_travel, &yaw, &t)) return;
    LOG(INFO) << "GNSS aligned on " << align_locals_.size() << " SPP fixes: yaw " << yaw * R2D << " deg, t_ecef_enu "
              << t.transpose();
    // step 3: the yaw from the TDCP displacements; t_ecef_enu then matches the centroids again under that yaw
    if (param->ali_refine && RefineYawTdcp(R_ecef_enu_, align_window_, param->tdcp_std, &yaw)) {
        Vec3 mean_l = Vec3::Zero(), mean_e = Vec3::Zero();
        for (size_t i = 0; i < align_locals_.size(); ++i) {
            mean_l += align_locals_[i];
            mean_e += align_ecefs_[i];
        }
        mean_l /= static_cast<double>(align_locals_.size());
        mean_e /= static_cast<double>(align_ecefs_.size());
        t = mean_e - R_ecef_enu_ * Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix() * mean_l;
        LOG(INFO) << "GNSS alignment refined on " << align_window_.size() << " epochs: yaw " << yaw * R2D
                  << " deg, t_ecef_enu " << t.transpose();
    }
    std::vector<Vec3>().swap(align_locals_);
    std::vector<Vec3>().swap(align_ecefs_);
    GnssAlignWindow().swap(align_window_);

    constexpr int kGnss = FastLioState::DIM - YAW;
    x.cov.middleRows<kGnss>(YAW).setZero();
    x.cov.middleCols<kGnss>(YAW).setZero();
    x.cov(YAW, YAW) = param->std_ext_yaw_init * param->std_ext_yaw_init;
    x.cov.block<3, 3>(TEC, TEC) = param->std_ext_pos_init * param->std_ext_pos_init * Mat3::Identity();
    x.cov.block<4, 4>(CLK, CLK) = param->std_clk_init * param->std_clk_init * Eigen::Matrix4d::Identity();
    x.yaw = yaw;
    x.t_ecef_enu = t;
    x.clk = epoch.clk_spp;
    gnss_aligned_ = true;
}

}  // namespace kuma_lio
