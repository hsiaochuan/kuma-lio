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

#include "local_ba.h"

#include "timer.h"

#include <glog/logging.h>
#include <algorithm>
#include <Eigen/Eigenvalues>

namespace kuma_lio {
namespace {
using StateVec = std::vector<LbaState, Eigen::aligned_allocator<LbaState>>;

Mat3 RightJacobian(const Vec3& phi) {
    Mat3 J;
    rightJacobianSO3(phi, J);
    return J;
}
Mat3 RightJacobianInv(const Vec3& phi) {
    Mat3 J;
    rightJacobianInvSO3(phi, J);
    return J;
}
// A plane only explains points within this many radii of its centre.
constexpr double kPlaneRadiusK = 3.0;

// Whether the plane (`center`, `normal`, `radius`, covariance `plane_var` of (normal, center))
// explains `pv`, which lies `dist` off it: close enough to the centre, and within `sigma_num`
// standard deviations.  Fills `hit` except for `d`; `sigma_l` is the variance of `dist`.
bool GatePlane(double dist, const Vec3& center, const Vec3& normal, double radius,
                      const Eigen::Matrix<double, 6, 6>& plane_var, const LbaPoint& pv, double sigma_num,
                      PlaneHit* hit, double* sigma_l) {
    const Vec3& p = pv.p;
    const double range_dis = std::sqrt(std::max((center - p).squaredNorm() - dist * dist, 0.0));
    if (range_dis > kPlaneRadiusK * radius) return false;
    Eigen::Matrix<double, 1, 6> J_nq;
    J_nq.block<1, 3>(0, 0) = p - center;
    J_nq.block<1, 3>(0, 3) = -normal;
    const double sigma_plane = J_nq * plane_var * J_nq.transpose();
    *sigma_l = sigma_plane + normal.dot(pv.var * normal);
    if (std::abs(dist) >= sigma_num * std::sqrt(*sigma_l)) return false;
    hit->normal = normal;
    hit->signed_dist = dist;
    hit->plane_sigma = sigma_plane;
    return true;
}

// Voxel-SLAM Bf_var: covariance of the contribution of a point `pw` with covariance `var` to (P, v).
void AddPointCov(const Vec3& pw, const Mat3& var, Eigen::Matrix<double, 9, 9>* cov) {
    Eigen::Matrix<double, 6, 3> Bi;
    Bi << 2 * pw(0), 0, 0, pw(1), pw(0), 0, pw(2), 0, pw(0), 0, 2 * pw(1), 0, 0, pw(2), pw(1), 0, 0, 2 * pw(2);
    const Eigen::Matrix<double, 6, 3> Biup = Bi * var;
    cov->block<6, 6>(0, 0) += Biup * Bi.transpose();
    cov->block<6, 3>(0, 6) += Biup;
    cov->block<3, 6>(6, 0) += Biup.transpose();
    cov->block<3, 3>(6, 6) += var;
}
}  // namespace

Mat3 BodyPointCov(const Vec3& p_body, double range_err, double angle_err) {
    Vec3 pb = p_body;
    if (pb[2] == 0) pb[2] = 0.0001;
    const double range = pb.norm();
    const double range_var = range_err * range_err;
    const double s = std::sin(angle_err * M_PI / 180.0);
    const Vec3 dir = pb / range;
    Mat3 dir_hat;
    dir_hat << 0, -dir(2), dir(1), dir(2), 0, -dir(0), -dir(1), dir(0), 0;
    Vec3 b1(1, 1, -(dir(0) + dir(1)) / dir(2));
    b1.normalize();
    Vec3 b2 = b1.cross(dir);
    b2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N.col(0) = b1;
    N.col(1) = b2;
    const Eigen::Matrix<double, 3, 2> A = range * dir_hat * N;
    return dir * range_var * dir.transpose() + s * s * A * A.transpose();
}

void ImuPreint::Add(const Vec3& gyro, const Vec3& acc, double dt) {
    const Vec3 w = gyro - bg_;
    const Vec3 a = acc - ba_;
    dtime_ += dt;
    const Mat3 R_inc = ExpMat(w * dt);
    const Mat3 R_jr = RightJacobian(w * dt);
    const Mat3 R_dt = dt * R_delta_;
    const Mat3 R_dt2_2 = 0.5 * dt * dt * R_delta_;
    const Mat3 acc_skew = Hat(a);

    Mat9 A = Mat9::Identity();
    Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
    A.block<3, 3>(0, 0) = R_inc.transpose();
    A.block<3, 3>(3, 0) = -R_dt2_2 * acc_skew;
    A.block<3, 3>(3, 6) = Mat3::Identity() * dt;
    A.block<3, 3>(6, 0) = -R_dt * acc_skew;
    B.block<3, 3>(0, 0) = R_jr * dt;
    B.block<3, 3>(3, 3) = R_dt2_2;
    B.block<3, 3>(6, 3) = R_dt;
    cov_ = A * cov_ * A.transpose() + B * noise_meas_ * B.transpose();
    cov_walk_ += noise_walk_ * dt;

    // first-order dependence of the deltas on the biases, before the deltas move on
    p_ba_ += v_ba_ * dt - R_dt2_2;
    p_bg_ += v_bg_ * dt - R_dt2_2 * acc_skew * R_bg_;
    v_ba_ -= R_dt;
    v_bg_ -= R_dt * acc_skew * R_bg_;
    R_bg_ = R_inc.transpose() * R_bg_ - R_jr * dt;

    p_delta_ += v_delta_ * dt + R_dt2_2 * a;
    v_delta_ += R_dt * a;
    R_delta_ = R_delta_ * R_inc;
}

// Voxel-SLAM IMU_PRE::give_evaluate
double ImuPreint::Evaluate(const LbaState& s1, const LbaState& s2, const Vec3& g, Mat30* jtj, Vec30* gg) const {
    using Mat15 = Eigen::Matrix<double, 15, 15>;
    const Vec3 dbg = s1.bg - bg_, dba = s1.ba - ba_;
    const Mat3 R_corr = R_delta_ * ExpMat(Vec3(R_bg_ * dbg));
    const Vec3 p_corr = p_delta_ + p_bg_ * dbg + p_ba_ * dba;
    const Vec3 v_corr = v_delta_ + v_bg_ * dbg + v_ba_ * dba;
    const Mat3 res_r = R_corr.transpose() * s1.R.transpose() * s2.R;
    const Vec3 exp_v = s1.R.transpose() * (s2.v - s1.v - dtime_ * g);
    const Vec3 exp_t = s1.R.transpose() * (s2.p - s1.p - s1.v * dtime_ - 0.5 * dtime_ * dtime_ * g);
    const Vec3 log_r = LogMat(res_r);

    Eigen::Matrix<double, 15, 1> rr;
    rr << log_r, exp_t - p_corr, exp_v - v_corr, s2.bg - s1.bg, s2.ba - s1.ba;
    Mat15 cov = Mat15::Zero();
    cov.topLeftCorner<9, 9>() = cov_;
    cov.bottomRightCorner<6, 6>() = cov_walk_;
    const Mat15 cov_inv = cov.inverse();

    if (jtj) {
        Eigen::Matrix<double, 15, 30> joc = Eigen::Matrix<double, 15, 30>::Zero();
        const Mat3 JR_inv = RightJacobianInv(log_r);
        const Mat3 R1T = s1.R.transpose();
        // s1
        joc.block<3, 3>(0, 0) = -JR_inv * s2.R.transpose() * s1.R;
        joc.block<3, 3>(0, 9) = -JR_inv * res_r.transpose() * RightJacobian(Vec3(R_bg_ * dbg)) * R_bg_;
        joc.block<3, 3>(3, 0) = Hat(exp_t);
        joc.block<3, 3>(3, 3) = -R1T;
        joc.block<3, 3>(3, 6) = -R1T * dtime_;
        joc.block<3, 3>(3, 9) = -p_bg_;
        joc.block<3, 3>(3, 12) = -p_ba_;
        joc.block<3, 3>(6, 0) = Hat(exp_v);
        joc.block<3, 3>(6, 6) = -R1T;
        joc.block<3, 3>(6, 9) = -v_bg_;
        joc.block<3, 3>(6, 12) = -v_ba_;
        joc.block<6, 6>(9, 9) = -Eigen::Matrix<double, 6, 6>::Identity();
        // s2
        joc.block<3, 3>(0, 15) = JR_inv;
        joc.block<3, 3>(3, 18) = R1T;
        joc.block<3, 3>(6, 21) = R1T;
        joc.block<6, 6>(9, 24) = Eigen::Matrix<double, 6, 6>::Identity();

        *jtj = joc.transpose() * cov_inv * joc;
        *gg = joc.transpose() * cov_inv * rr;
    }
    return rr.dot(cov_inv * rr);
}

LocalBA::LocalBA(const LbaParam& param) : param_(param), g_(Vec3::Zero()) {
    CHECK_GE(param_.win_size, 2);
    noise_meas_.setZero();
    noise_meas_.diagonal() << Vec3::Constant(param_.cov_gyr), Vec3::Constant(param_.cov_acc);
    noise_walk_.setZero();
    noise_walk_.diagonal() << Vec3::Constant(param_.rdw_gyr), Vec3::Constant(param_.rdw_acc);
}

void LocalBA::AddImu(int64_t time_ns, const Vec3& gyro, const Vec3& acc) {
    Imu imu;
    imu.time_ns = time_ns;
    imu.angular_velocity = gyro;
    imu.linear_acceleration = acc;
    imu_buf_.push_back(imu);
}

bool LocalBA::AddFrame(int64_t time_ns, const LbaState& odom, const std::vector<LbaPoint>& points,
                       const Vec3& gravity) {
    g_ = -gravity;

    Frame frame;
    frame.id = frames_.empty() ? 0 : frames_.back().id + 1;
    frame.time_ns = time_ns;
    frame.x = odom;
    if (!frames_.empty()) {
        const Frame& prev = frames_.back();
        // midpoint integration
        frame.imu = std::make_shared<ImuPreint>(prev.x.bg, prev.x.ba, noise_meas_, noise_walk_);
        CHECK(!imu_buf_.empty()) << "local BA frame without IMU";
        while (imu_buf_.size() > 1 && imu_buf_[1].time_ns <= prev.time_ns)
            imu_buf_.pop_front();
        int64_t cursor = prev.time_ns;
        for (size_t i = 0; i < imu_buf_.size() && cursor < time_ns; ++i) {
            const Imu& b = imu_buf_[i];
            const Imu& a = i > 0 ? imu_buf_[i - 1] : b;
            const int64_t until = std::min(b.time_ns, time_ns);
            if (until <= cursor) continue;
            frame.imu->Add(0.5 * (a.angular_velocity + b.angular_velocity),
                           0.5 * (a.linear_acceleration + b.linear_acceleration), NS_TO_S(until - cursor));
            cursor = until;
        }
        if (cursor < time_ns) {
            frame.imu->Add(imu_buf_.back().angular_velocity, imu_buf_.back().linear_acceleration,
                           NS_TO_S(time_ns - cursor));
        }
    }
    frames_.push_back(std::move(frame));
    Frame& cur = frames_.back();
    Timer timer;
    InsertFrame(cur, points);
    const double t_insert = timer.ElapsedMiniSeconds(true);

    Refresh(true);
    const double t_refresh = timer.ElapsedMiniSeconds(true);
    const bool full = static_cast<int>(frames_.size()) >= param_.win_size;
    if (full) Optimize();
    const double t_optimize = timer.ElapsedMiniSeconds(true);
    if (full) {
        Marginalize(frames_.front());
        frames_.pop_front();
        Refresh();  // the planes the odometry sees follow the refined poses
        if (cur.id % 50 == 0) Slide(frames_.back().x.p);
    }
    LOG(INFO) << "LBA time [ms]: insert " << t_insert << " refresh " << t_refresh << " optimize " << t_optimize
              << " marginalize+refresh " << timer.ElapsedMiniSeconds() << " points " << points.size() << " active "
              << active_.size();
    return full;
}

LocalBA::Cell& LocalBA::CellAt(const Vec3& pw) {
    const VOXEL_LOCATION loc(pw, param_.voxel_size);
    const int idx = CellIndex(pw, loc);
    Root& root = map_[loc];
    root.dirty = true;
    if (!root.active) {
        root.active = true;
        active_.push_back(&root);
    }
    std::vector<Cell>& cells = root.cells;
    auto cell = std::find_if(cells.begin(), cells.end(), [idx](const Cell& k) { return k.idx == idx; });
    if (cell == cells.end()) {
        cells.emplace_back();
        cell = cells.end() - 1;
        cell->idx = idx;
    }
    return *cell;
}

void LocalBA::InitMap(const std::vector<Vec3>& points, double var) {
    for (const Vec3& pw : points) {
        Cell& cell = CellAt(pw);
        if (cell.fix.N >= param_.max_fix_points) continue;
        cell.fix.Push(pw);
        AddPointCov(pw, var * Mat3::Identity(), &cell.fix_cov);
    }
    Refresh();
}

void LocalBA::InsertFrame(const Frame& frame, const std::vector<LbaPoint>& points) {
    for (const LbaPoint& pt : points) {
        const Vec3 pw = frame.x.R * pt.p + frame.x.p;
        const Mat3 var = frame.x.R * pt.var * frame.x.R.transpose() + param_.pose_var * Mat3::Identity();
        Cell& cell = CellAt(pw);
        if (cell.clusters.empty() || cell.clusters.back().id != frame.id) {
            cell.clusters.emplace_back();
            cell.clusters.back().id = frame.id;
        }
        cell.clusters.back().clu.Push(pt.p);
        AddPointCov(pw, var, &cell.clusters.back().cov);
    }
}

//a node of layer l is looked at only when its parent at l - 1 held enough points and was not flat.
void LocalBA::Refresh(bool dirty_only) {
    const int W = static_cast<int>(frames_.size());
    // a window of one frame has no factors, so there are none to reuse in the next
    if (W <= 2) dirty_only = false;
    const int num_roots = static_cast<int>(active_.size());
#pragma omp parallel num_threads(param_.num_threads) if (num_roots > 256)
    {
        RefitScratch scratch;
        scratch.win.resize(kCells * W);
        scratch.fix.resize(kCells);
        scratch.all.resize(kCells);
        scratch.cov.resize(kCells);
#pragma omp for schedule(dynamic, 32)
        for (int i = 0; i < num_roots; ++i) {
            Root& root = *active_[i];
            if (dirty_only && !root.dirty) continue;
            RefitRoot(&root, &scratch);
            root.dirty = false;
        }
    }
    factors_.clear();
    std::vector<Root*> active_voxels;
    for (Root* root : active_) {
        if (root->active) active_voxels.push_back(root);
        for (PlaneFactor& f : root->factors) {
            f.clus.resize(W);  // a factor kept from before the newest frame was added
            factors_.push_back(&f);
        }
    }
    active_.swap(active_voxels);
}

void LocalBA::RefitRoot(Root* root_ptr, RefitScratch* scratch) const {
    const int W = static_cast<int>(frames_.size());
    const int base = frames_.empty() ? 0 : frames_.front().id;
    std::vector<PointCluster>& nodes_clus = scratch->win;
    std::vector<PointCluster>& nodes_fix = scratch->fix;
    std::vector<PointCluster>& nodes_total = scratch->all;
    Eigen::aligned_vector<Mat9>& nodes_cov = scratch->cov;
    PointCluster tr_clu;
    Root& root = *root_ptr;
    root.active = std::any_of(root.cells.begin(), root.cells.end(), [](const Cell& c) { return !c.clusters.empty(); });
    root.planes.clear();
    root.factors.clear();
    bool split[kLayers][kCells] = {};
    bool on_plane[kLayers][kCells] = {};
    for (int layer = 0; layer < kLayers; ++layer) {
        const int num_nodes = 1 << (3 * layer);
        bool touched[kCells] = {};
        bool any = false;
        for (const Cell& cell : root.cells) {
            if (layer > 0 && !split[layer - 1][NodeOf(cell.idx, layer - 1)]) continue;
            const int n = NodeOf(cell.idx, layer);
            if (!touched[n]) {
                touched[n] = any = true;
                std::fill(nodes_clus.begin() + n * W, nodes_clus.begin() + (n + 1) * W, PointCluster());
                nodes_fix[n] = PointCluster();
                nodes_total[n] = PointCluster();
                nodes_cov[n].setZero();
            }
            nodes_fix[n] += cell.fix;
            nodes_total[n] += cell.fix;
            nodes_cov[n] += cell.fix_cov;
            for (const FrameCluster& clu : cell.clusters) {
                const LbaState& x = frames_[clu.id - base].x;
                nodes_clus[n * W + clu.id - base] += clu.clu;
                tr_clu.Transform(clu.clu, x.R, x.p);
                nodes_total[n] += tr_clu;
                nodes_cov[n] += clu.cov;
            }
        } // for cell
        if (!any) break;
        for (int n = 0; n < num_nodes; ++n) {
            if (!touched[n] || nodes_total[n].N <= param_.min_points) continue;
            const Eigen::SelfAdjointEigenSolver<Mat3> saes(nodes_total[n].Cov());
            const Vec3 eig = saes.eigenvalues();
            // lambda_0 / lambda_1
            if (eig[0] >= param_.plane_eigen_ratio * eig[1]) {
                split[layer][n] = true;
                continue;
            }
            // extent
            const double extent = param_.min_plane_size * param_.voxel_size / (1 << layer);
            if (eig[1] < extent * extent) continue;
            const Mat3 U = saes.eigenvectors();
            Plane plane;
            plane.layer = layer;
            plane.node = n;
            plane.center = nodes_total[n].Mean();
            plane.normal = U.col(0);
            plane.radius = std::sqrt(eig[2]);
            const double nv = 1.0 / nodes_total[n].N;
            Eigen::Matrix<double, 3, 9> u_c = Eigen::Matrix<double, 3, 9>::Zero();
            for (int k = 1; k < 3; ++k) {
                const Mat3 ukl = U.col(k) * U.col(0).transpose();
                Eigen::Matrix<double, 1, 9> fkl;
                fkl.head<6>() << ukl(0, 0), ukl(1, 0) + ukl(0, 1), ukl(2, 0) + ukl(0, 2), ukl(1, 1),
                    ukl(1, 2) + ukl(2, 1), ukl(2, 2);
                fkl.tail<3>() =
                    -(U.col(k).dot(plane.center) * U.col(0) + U.col(0).dot(plane.center) * U.col(k)).transpose();
                u_c += nv / (eig[0] - eig[k]) * U.col(k) * fkl;
            }
            const Eigen::Matrix<double, 3, 9> Jc = u_c * nodes_cov[n];
            plane.plane_var.block<3, 3>(0, 0) = Jc * u_c.transpose();
            plane.plane_var.block<3, 3>(0, 3) = nv * Jc.block<3, 3>(0, 6);
            plane.plane_var.block<3, 3>(3, 0) = plane.plane_var.block<3, 3>(0, 3).transpose();
            plane.plane_var.block<3, 3>(3, 3) = nv * nv * nodes_cov[n].block<3, 3>(6, 6);
            if (!plane.plane_var.allFinite()) continue;
            root.planes.push_back(plane);
            on_plane[layer][n] = true;

            if (W >= 2 && nodes_total[n].N > nodes_fix[n].N) {
                PlaneFactor factor;
                factor.fix = nodes_fix[n];
                factor.clus.assign(nodes_clus.begin() + n * W, nodes_clus.begin() + (n + 1) * W);
                root.factors.push_back(std::move(factor));
            }
        }
    } // for layer
    if (param_.clutter_points > 0) {
        for (Cell& cell : root.cells) {
            bool on = false;
            for (int l = 0; l < kLayers && !on; ++l) on = on_plane[l][NodeOf(cell.idx, l)];
            if (on) continue;
            int cnt = cell.fix.N;
            for (const FrameCluster& clu : cell.clusters)
                cnt += clu.clu.N;
            if (cnt < param_.clutter_points) continue;
            cell.fix = PointCluster();
            cell.fix_cov.setZero();
            std::vector<FrameCluster>().swap(cell.clusters);
        }
    }
    if (!root.active) {
        std::vector<PlaneFactor>().swap(root.factors);
        root.planes.shrink_to_fit();
    }
}

void LocalBA::Marginalize(const Frame& frame) {
    PointCluster world;
    for (Root* root : active_) {
        for (Cell& cell : root->cells) {
            if (cell.clusters.empty() || cell.clusters.front().id != frame.id) continue;
            if (cell.fix.N < param_.max_fix_points) {
                world.Transform(cell.clusters.front().clu, frame.x.R, frame.x.p);
                cell.fix += world;
                cell.fix_cov += cell.clusters.front().cov;
            }
            cell.clusters.erase(cell.clusters.begin());
            // most cells leave the window for good: give the capacity back (100k voxels on a long run)
            if (cell.clusters.empty()) std::vector<FrameCluster>().swap(cell.clusters);
        }
    }
}

void LocalBA::Slide(const Vec3& center) {
    // a voxel still seen from the window is within sensor range, so only marginalized ones go
    const double half = 0.5 * param_.voxel_size;
    for (auto it = map_.begin(); it != map_.end();) {
        const VOXEL_LOCATION& l = it->first;
        const Vec3 c = Vec3(l.x, l.y, l.z) * param_.voxel_size + Vec3::Constant(half);
        if (!it->second.active && (c - center).norm() > param_.map_range) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

bool LocalBA::MatchRoot(const VOXEL_LOCATION& loc, const LbaPoint& pv, PlaneHit* hit) const {
    const auto it = map_.find(loc);
    if (it == map_.end()) return false;
    const Vec3& p = pv.p;
    const int idx = CellIndex(p, loc);
    for (const Plane& plane : it->second.planes) {
        if (plane.node != NodeOf(idx, plane.layer)) continue;
        // at most one node on the way down to the cell is a plane
        const double dist = plane.normal.dot(p - plane.center);
        double sigma_l;
        if (!GatePlane(dist, plane.center, plane.normal, plane.radius, plane.plane_var, pv, param_.sigma_num, hit,
                       &sigma_l)) {
            return false;
        }
        hit->d = -plane.normal.dot(plane.center);
        return true;
    }
    return false;
}

bool LocalBA::Query(const LbaPoint& pv, PlaneHit* hit) const {
    const VOXEL_LOCATION loc(pv.p, param_.voxel_size);
    if (MatchRoot(loc, pv, hit)) return true;
    // near a voxel face the plane may live in the neighbour
    VOXEL_LOCATION near = loc;
    const Vec3 rel = pv.p / param_.voxel_size - Vec3(loc.x, loc.y, loc.z);  // in [0, 1)
    if (rel[0] > 0.75) near.x++; else if (rel[0] < 0.25) near.x--;
    if (rel[1] > 0.75) near.y++; else if (rel[1] < 0.25) near.y--;
    if (rel[2] > 0.75) near.z++; else if (rel[2] < 0.25) near.z--;
    if (near == loc) return false;
    return MatchRoot(near, pv, hit);
}

bool LocalBA::PlaneNormal(const Vec3& p, Vec3* normal) const {
    const VOXEL_LOCATION loc(p, param_.voxel_size);
    const auto it = map_.find(loc);
    if (it == map_.end()) return false;
    const int idx = CellIndex(p, loc);
    for (const Plane& plane : it->second.planes) {
        if (plane.node != NodeOf(idx, plane.layer)) continue;
        *normal = plane.normal;
        return true;
    }
    return false;
}

// Voxel-SLAM LidarFactor::acc_evaluate2: the cost of a plane is the smallest eigenvalue of the
// covariance of all its points, with the closed-form gradient and Hessian over the window poses.
// Parallel in two passes that keep the serial result bit for bit: the per-plane terms are computed
// independently, then every Hessian block (and gradient segment) is summed over the planes in their
// original order, so each element sees the same sequence of additions as a single loop would.
double LocalBA::EvaluateLidar(const StateVec& xs, Eigen::MatrixXd* hess, Eigen::VectorXd* jact) const {
    const int W = static_cast<int>(xs.size());
    const int F = static_cast<int>(factors_.size());
    const int threads = F > 256 ? param_.num_threads : 1;
    std::vector<double> lambda0(F);
    // per plane: ukukT, umumT; per plane and window frame: Auk, viRiTuk, viRiTukukT, the gradient and the diagonal block
    std::vector<double> NNs(hess ? F : 0);
    Eigen::aligned_vector<Mat3> ukukTs(hess ? F : 0), umumTs(hess ? F : 0);
    Eigen::aligned_vector<Eigen::Matrix<double, 3, 6>> Auks(hess ? F * W : 0);
    Eigen::aligned_vector<Vec3> viRiTuks(hess ? F * W : 0);
    Eigen::aligned_vector<Mat3> viRiTukukTs(hess ? F * W : 0);
    Eigen::aligned_vector<Eigen::Matrix<double, 6, 1>> jjts(hess ? F * W : 0);
    Eigen::aligned_vector<Eigen::Matrix<double, 6, 6>> Hdiags(hess ? F * W : 0);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int k = 0; k < F; ++k) {
        const PlaneFactor& factor = *factors_[k];
        PointCluster tr_clu;
        PointCluster total = factor.fix;
        for (int i = 0; i < W; ++i) {
            if (factor.clus[i].N == 0) continue;
            tr_clu.Transform(factor.clus[i], xs[i].R, xs[i].p);
            total += tr_clu;
        }
        const double NN = total.N;
        const Vec3 vBar = total.v / NN;
        const Eigen::SelfAdjointEigenSolver<Mat3> saes(total.P / NN - vBar * vBar.transpose());
        const Vec3 lmbd = saes.eigenvalues();
        lambda0[k] = lmbd[0];
        if (!hess) continue;

        const Mat3 U = saes.eigenvectors();
        const Vec3 uk = U.col(0);
        const Mat3 ukukT = uk * uk.transpose();
        Mat3 umumT = Mat3::Zero();
        for (int i = 1; i < 3; ++i) umumT += 2.0 / (lmbd[0] - lmbd[i]) * U.col(i) * U.col(i).transpose();
        NNs[k] = NN;
        ukukTs[k] = ukukT;
        umumTs[k] = umumT;

        for (int i = 0; i < W; ++i) {
            if (factor.clus[i].N == 0) continue;
            const Mat3& Pi = factor.clus[i].P;
            const Vec3& vi = factor.clus[i].v;
            const Mat3& Ri = xs[i].R;
            const double ni = factor.clus[i].N;
            Vec3& viRiTuk = viRiTuks[k * W + i];
            Mat3& viRiTukukT = viRiTukukTs[k * W + i];
            Eigen::Matrix<double, 3, 6>& Auk = Auks[k * W + i];

            const Mat3 vihat = Hat(vi);
            const Vec3 RiTuk = Ri.transpose() * uk;
            const Mat3 RiTukhat = Hat(RiTuk);
            const Vec3 PiRiTuk = Pi * RiTuk;
            viRiTuk = vihat * RiTuk;
            viRiTukukT = viRiTuk * uk.transpose();

            const Vec3 ti_v = xs[i].p - vBar;
            const double ukTti_v = uk.dot(ti_v);
            const Mat3 combo1 = Hat(PiRiTuk) + vihat * ukTti_v;
            const Vec3 combo2 = Ri * vi + ni * ti_v;
            Auk.block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
            Auk.block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * Mat3::Identity();
            Auk /= NN;

            const Eigen::Matrix<double, 6, 1> jjt = Auk.transpose() * uk;
            jjts[k * W + i] = jjt;

            const Mat3 HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT;
            Eigen::Matrix<double, 6, 6> Hb = Auk.transpose() * umumT * Auk;
            Hb.block<3, 3>(0, 0) += 2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat -
                                    2.0 / NN / NN * viRiTuk * viRiTuk.transpose() - 0.5 * Hat(jjt.head<3>());
            Hb.block<3, 3>(0, 3) += HRt;
            Hb.block<3, 3>(3, 0) += HRt.transpose();
            Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;
            Hdiags[k * W + i] = Hb;
        }
    }
    double residual = 0.0;
    for (int k = 0; k < F; ++k) residual += lambda0[k];
    if (!hess) return residual;

    // one task per upper-triangle block (i, j); the diagonal ones also carry the gradient segment
    std::vector<std::pair<int, int>> blocks;
    for (int i = 0; i < W; ++i)
        for (int j = i; j < W; ++j) blocks.emplace_back(i, j);
    const int num_blocks = static_cast<int>(blocks.size());
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (int b = 0; b < num_blocks; ++b) {
        const int i = blocks[b].first, j = blocks[b].second;
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        if (i == j) {
            Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
            for (int k = 0; k < F; ++k) {
                if (factors_[k]->clus[i].N == 0) continue;
                g += jjts[k * W + i];
                H += Hdiags[k * W + i];
            }
            jact->segment<6>(kDim * i) += g;
        } else {
            for (int k = 0; k < F; ++k) {
                const PlaneFactor& f = *factors_[k];
                if (f.clus[i].N == 0 || f.clus[j].N == 0) continue;
                const double NN = NNs[k];
                const double ni = f.clus[i].N;
                const double nj = f.clus[j].N;
                const Eigen::Matrix<double, 3, 6>& Auki = Auks[k * W + i];
                const Eigen::Matrix<double, 3, 6>& Aukj = Auks[k * W + j];
                Eigen::Matrix<double, 6, 6> Hb = Auki.transpose() * umumTs[k] * Aukj;
                Hb.block<3, 3>(0, 0) += -2.0 / NN / NN * viRiTuks[k * W + i] * viRiTuks[k * W + j].transpose();
                Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukTs[k * W + i];
                Hb.block<3, 3>(3, 0) += -2.0 * ni / NN / NN * viRiTukukTs[k * W + j].transpose();
                Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukTs[k];
                H += Hb;
            }
        }
        // only the upper triangle was summed
        hess->block<6, 6>(kDim * i, kDim * j) += H;
        if (j != i) hess->block<6, 6>(kDim * j, kDim * i) += H.transpose();
    }
    return residual;
}

double LocalBA::Evaluate(const StateVec& xs, Eigen::MatrixXd* hess, Eigen::VectorXd* jact) const {
    const int W = static_cast<int>(xs.size());
    double residual = 0.0;
    if (hess) {
        hess->setZero();
        jact->setZero();
    }
    ImuPreint::Mat30 jtj;
    ImuPreint::Vec30 gg;
    for (int i = 0; i + 1 < W; ++i) {
        residual += frames_[i + 1].imu->Evaluate(xs[i], xs[i + 1], g_, hess ? &jtj : nullptr, hess ? &gg : nullptr);
        if (hess) {
            hess->block<2 * kDim, 2 * kDim>(kDim * i, kDim * i) += jtj;
            jact->segment<2 * kDim>(kDim * i) += gg;
        }
    }
    residual *= 0.5 * param_.imu_coef;
    if (hess) {
        *hess *= param_.imu_coef;
        *jact *= param_.imu_coef;
    }
    residual += EvaluateLidar(xs, hess, jact);
    return residual;
}

// Voxel-SLAM LI_BA_Optimizer::damping_iter: Levenberg-Marquardt with the whole oldest state (R p v bg ba) held fixed.
void LocalBA::Optimize() {
    const int W = static_cast<int>(frames_.size());
    const int n = kDim * W;
    StateVec xs(W), xs_temp(W);
    for (int i = 0; i < W; ++i) xs[i] = frames_[i].x;

    double u = 0.01, v = 2;
    Eigen::MatrixXd Hess(n, n), D = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd JacT(n), dxi(n);
    double residual1 = 0.0, residual2 = 0.0, residual0 = 0.0;
    bool is_calc_hess = true;
    for (int iter = 0; iter < param_.max_iter; ++iter) {
        if (is_calc_hess) {
            residual1 = Evaluate(xs, &Hess, &JacT);
            if (iter == 0)
                residual0 = residual1;
            Hess.topRows(kDim).setZero();
            Hess.leftCols(kDim).setZero();
            Hess.block<kDim, kDim>(0, 0).setIdentity();
            JacT.head(kDim).setZero();
        }
        D.diagonal() = Hess.diagonal();
        dxi = (Hess + u * D).ldlt().solve(-JacT);
        for (int j = 0; j < W; ++j) {
            xs_temp[j].R = xs[j].R * ExpMat(dxi.segment<3>(kDim * j));
            xs_temp[j].p = xs[j].p + dxi.segment<3>(kDim * j + 3);
            xs_temp[j].v = xs[j].v + dxi.segment<3>(kDim * j + 6);
            xs_temp[j].bg = xs[j].bg + dxi.segment<3>(kDim * j + 9);
            xs_temp[j].ba = xs[j].ba + dxi.segment<3>(kDim * j + 12);
        }
        const double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);
        residual2 = Evaluate(xs_temp, nullptr, nullptr);
        double q = residual1 - residual2;
        if (q > 0) {
            xs = xs_temp;
            q = q / q1;
            v = 2;
            q = 1 - std::pow(2 * q - 1, 3);
            u *= (q < 1.0 / 3 ? 1.0 / 3 : q);
            is_calc_hess = true;
        } else {
            u = u * v;
            v = 2 * v;
            is_calc_hess = false;
        }
        if (std::abs((residual1 - residual2) / residual1) < 1e-6) break;
    }
    const double moved = (xs.back().p - frames_.back().x.p).norm();
    for (int i = 0; i < W; ++i) frames_[i].x = xs[i];
    LOG(INFO) << "LBA: frame " << frames_.back().id << " planes " << factors_.size() << " cost " << residual0 << " -> "
              << std::min(residual1, residual2) << " moved " << moved << " voxels " << map_.size() << " bg "
              << xs.back().bg.transpose() << " ba " << xs.back().ba.transpose();
}

}  // namespace kuma_lio
