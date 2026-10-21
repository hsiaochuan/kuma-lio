#include "local_ba.h"

#include "timer.h"

#include <glog/logging.h>
#include <algorithm>
#include <Eigen/Eigenvalues>

namespace faster_lio {
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
/// A plane only explains points within this many radii of its centre.
constexpr double kPlaneRadiusK = 3.0;

/// Whether the plane (`center`, `normal`, `radius`, covariance `plane_var` of (normal, center))
/// explains `pv`, which lies `dist` off it: close enough to the centre, and within `sigma_num`
/// standard deviations.  Fills `hit` except for `d`; `sigma_l` is the variance of `dist`.
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

    p_delta_ += v_delta_ * dt + R_dt2_2 * a;
    v_delta_ += R_dt * a;
    R_delta_ = R_delta_ * R_inc;
}

double ImuPreint::Evaluate(const LbaState& s1, const LbaState& s2, const Vec3& g, Mat18* jtj, Vec18* gg) const {
    const Mat3 res_r = R_delta_.transpose() * s1.R.transpose() * s2.R;
    const Vec3 exp_v = s1.R.transpose() * (s2.v - s1.v - dtime_ * g);
    const Vec3 exp_t = s1.R.transpose() * (s2.p - s1.p - s1.v * dtime_ - 0.5 * dtime_ * dtime_ * g);
    const Vec3 log_r = LogMat(res_r);

    Eigen::Matrix<double, 9, 1> rr;
    rr.segment<3>(0) = log_r;
    rr.segment<3>(3) = exp_t - p_delta_;
    rr.segment<3>(6) = exp_v - v_delta_;
    const Mat9 cov_inv = cov_.inverse();

    if (jtj) {
        Eigen::Matrix<double, 9, 18> joc = Eigen::Matrix<double, 9, 18>::Zero();
        const Mat3 JR_inv = RightJacobianInv(log_r);
        const Mat3 R1T = s1.R.transpose();
        // s1
        joc.block<3, 3>(0, 0) = -JR_inv * s2.R.transpose() * s1.R;
        joc.block<3, 3>(3, 0) = Hat(exp_t);
        joc.block<3, 3>(3, 3) = -R1T;
        joc.block<3, 3>(3, 6) = -R1T * dtime_;
        joc.block<3, 3>(6, 0) = Hat(exp_v);
        joc.block<3, 3>(6, 6) = -R1T;
        // s2
        joc.block<3, 3>(0, 9) = JR_inv;
        joc.block<3, 3>(3, 12) = R1T;
        joc.block<3, 3>(6, 15) = R1T;

        *jtj = joc.transpose() * cov_inv * joc;
        *gg = joc.transpose() * cov_inv * rr;
    }
    return rr.dot(cov_inv * rr);
}

LocalBA::LocalBA(const LbaParam& param) : param_(param), g_(Vec3::Zero()) {
    CHECK_GE(param_.win_size, 2);
    noise_meas_.setZero();
    noise_meas_.diagonal() << Vec3::Constant(param_.cov_gyr), Vec3::Constant(param_.cov_acc);
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
    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
    odom_pose.linear() = odom.R;
    odom_pose.translation() = odom.p;

    Frame frame;
    frame.id = static_cast<int>(history_.size());
    frame.time_ns = time_ns;
    frame.x = odom;
    if (!frames_.empty()) {
        // The odometry matched this frame against the very map the BA maintains, so its pose is
        // already in the BA's frame and is the initial value as it stands.
        const Frame& prev = frames_.back();

        // midpoint integration over (prev.time, time], holding the end samples where the buffer falls short
        frame.imu = std::make_shared<ImuPreint>(prev.x.bg, prev.x.ba, noise_meas_);
        CHECK(!imu_buf_.empty()) << "local BA frame without IMU";
        while (imu_buf_.size() > 1 && imu_buf_[1].time_ns <= prev.time_ns) imu_buf_.pop_front();
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
    History h;
    h.time_ns = time_ns;
    h.odom = odom_pose;
    history_.push_back(h);
    frames_.push_back(frame);
    Timer timer;
    InsertFrame(frame, points);
    const double t_insert = timer.ElapsedMiniSeconds(true);

    Refresh(true);
    const double t_refresh = timer.ElapsedMiniSeconds(true);
    const bool full = static_cast<int>(frames_.size()) >= param_.win_size;
    if (full) Optimize();
    const double t_optimize = timer.ElapsedMiniSeconds(true);
    for (const Frame& f : frames_) {
        history_[f.id].refined.linear() = f.x.R;
        history_[f.id].refined.translation() = f.x.p;
    }
    newest_ = frames_.back().x;
    if (full) {
        Marginalize(frames_.front());
        frames_.pop_front();
        Refresh();  // the planes the odometry sees follow the refined poses
        if (frame.id % 50 == 0) Slide(newest_.p);
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
        if (cell.win.empty() || cell.win.back().id != frame.id) {
            cell.win.emplace_back();
            cell.win.back().id = frame.id;
        }
        cell.win.back().pc.Push(pt.p);
        AddPointCov(pw, var, &cell.win.back().cov);
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
            f.win.resize(W);  // a factor kept from before the newest frame was added
            factors_.push_back(&f);
        }
    }
    active_.swap(active_voxels);
}

void LocalBA::RefitRoot(Root* root_ptr, RefitScratch* scratch) const {
    const int W = static_cast<int>(frames_.size());
    const int base = frames_.empty() ? 0 : frames_.front().id;
    std::vector<PointCluster>& win = scratch->win;
    std::vector<PointCluster>& fix = scratch->fix;
    std::vector<PointCluster>& all = scratch->all;
    Eigen::aligned_vector<Mat9>& cov = scratch->cov;
    PointCluster world;
    Root& root = *root_ptr;
    root.active = std::any_of(root.cells.begin(), root.cells.end(), [](const Cell& c) { return !c.win.empty(); });
    root.planes.clear();
    root.factors.clear();
    bool split[kLayers][kCells] = {};
    for (int layer = 0; layer < kLayers; ++layer) {
        const int num_nodes = 1 << (3 * layer);
        // only the nodes that cells fall into are summed, and a layer nothing was split into ends the walk
        bool touched[kCells] = {};
        bool any = false;
        for (const Cell& cell : root.cells) {
            if (layer > 0 && !split[layer - 1][NodeOf(cell.idx, layer - 1)]) continue;
            const int n = NodeOf(cell.idx, layer);
            if (!touched[n]) {
                touched[n] = any = true;
                std::fill(win.begin() + n * W, win.begin() + (n + 1) * W, PointCluster());
                fix[n] = PointCluster();
                all[n] = PointCluster();
                cov[n].setZero();
            }
            fix[n] += cell.fix;
            all[n] += cell.fix;
            cov[n] += cell.fix_cov;
            for (const WinPoints& w : cell.win) {
                const LbaState& x = frames_[w.id - base].x;
                win[n * W + w.id - base] += w.pc;
                world.Transform(w.pc, x.R, x.p);
                all[n] += world;
                cov[n] += w.cov;
            }
        }
        if (!any) break;
        for (int n = 0; n < num_nodes; ++n) {
            if (!touched[n] || all[n].N <= param_.min_points) continue;
            const Eigen::SelfAdjointEigenSolver<Mat3> saes(all[n].Cov());
            const Vec3 eig = saes.eigenvalues();
            if (eig[0] >= param_.min_eigen_value) {
                split[layer][n] = true;
                continue;
            }
            // plane_update: first-order covariance of (normal, center)
            const Mat3 U = saes.eigenvectors();
            Plane plane;
            plane.layer = layer;
            plane.node = n;
            plane.center = all[n].Mean();
            plane.normal = U.col(0);
            plane.radius = std::sqrt(eig[2]);
            const double nv = 1.0 / all[n].N;
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
            const Eigen::Matrix<double, 3, 9> Jc = u_c * cov[n];
            plane.plane_var.block<3, 3>(0, 0) = Jc * u_c.transpose();
            plane.plane_var.block<3, 3>(0, 3) = nv * Jc.block<3, 3>(0, 6);
            plane.plane_var.block<3, 3>(3, 0) = plane.plane_var.block<3, 3>(0, 3).transpose();
            plane.plane_var.block<3, 3>(3, 3) = nv * nv * cov[n].block<3, 3>(6, 6);
            if (!plane.plane_var.allFinite()) continue;
            root.planes.push_back(plane);

            if (W >= 2 && all[n].N > fix[n].N && eig[0] / eig[1] <= 0.12) {
                // a plane with points of the window on it (and not a line)
                PlaneFactor f;
                f.fix = fix[n];
                f.win.assign(win.begin() + n * W, win.begin() + (n + 1) * W);
                root.factors.push_back(std::move(f));
            }
        }
    } // for layer
}

void LocalBA::Marginalize(const Frame& frame) {
    PointCluster world;
    for (Root* root : active_) {
        for (Cell& cell : root->cells) {
            if (cell.win.empty() || cell.win.front().id != frame.id) continue;
            if (cell.fix.N < param_.max_fix_points) {
                world.Transform(cell.win.front().pc, frame.x.R, frame.x.p);
                cell.fix += world;
                cell.fix_cov += cell.win.front().cov;
            }
            cell.win.erase(cell.win.begin());
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
double LocalBA::EvaluateLidar(const StateVec& xs, Eigen::MatrixXd* hess, Eigen::VectorXd* jact) const {
    const int W = static_cast<int>(xs.size());
    double residual = 0.0;
    Eigen::MatrixXd H6;
    Eigen::VectorXd g6;
    if (hess) {
        H6 = Eigen::MatrixXd::Zero(6 * W, 6 * W);
        g6 = Eigen::VectorXd::Zero(6 * W);
    }
    Eigen::aligned_vector<Vec3> viRiTuk(W);
    Eigen::aligned_vector<Mat3> viRiTukukT(W);
    Eigen::aligned_vector<Eigen::Matrix<double, 3, 6>> Auk(W);
    PointCluster pcr;
    for (const PlaneFactor* factor : factors_) {
        const PlaneFactor& f = *factor;
        PointCluster sig = f.fix;
        for (int i = 0; i < W; ++i) {
            if (f.win[i].N == 0) continue;
            pcr.Transform(f.win[i], xs[i].R, xs[i].p);
            sig += pcr;
        }
        const double NN = sig.N;
        const Vec3 vBar = sig.v / NN;
        const Eigen::SelfAdjointEigenSolver<Mat3> saes(sig.P / NN - vBar * vBar.transpose());
        const Vec3 lmbd = saes.eigenvalues();
        residual += lmbd[0];
        if (!hess) continue;

        const Mat3 U = saes.eigenvectors();
        const Vec3 uk = U.col(0);
        const Mat3 ukukT = uk * uk.transpose();
        Mat3 umumT = Mat3::Zero();
        for (int i = 1; i < 3; ++i) umumT += 2.0 / (lmbd[0] - lmbd[i]) * U.col(i) * U.col(i).transpose();

        for (int i = 0; i < W; ++i) {
            if (f.win[i].N == 0) continue;
            const Mat3& Pi = f.win[i].P;
            const Vec3& vi = f.win[i].v;
            const Mat3& Ri = xs[i].R;
            const double ni = f.win[i].N;

            const Mat3 vihat = Hat(vi);
            const Vec3 RiTuk = Ri.transpose() * uk;
            const Mat3 RiTukhat = Hat(RiTuk);
            const Vec3 PiRiTuk = Pi * RiTuk;
            viRiTuk[i] = vihat * RiTuk;
            viRiTukukT[i] = viRiTuk[i] * uk.transpose();

            const Vec3 ti_v = xs[i].p - vBar;
            const double ukTti_v = uk.dot(ti_v);
            const Mat3 combo1 = Hat(PiRiTuk) + vihat * ukTti_v;
            const Vec3 combo2 = Ri * vi + ni * ti_v;
            Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
            Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * Mat3::Identity();
            Auk[i] /= NN;

            const Eigen::Matrix<double, 6, 1> jjt = Auk[i].transpose() * uk;
            g6.segment<6>(6 * i) += jjt;

            const Mat3 HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
            Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
            Hb.block<3, 3>(0, 0) += 2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat -
                                    2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5 * Hat(jjt.head<3>());
            Hb.block<3, 3>(0, 3) += HRt;
            Hb.block<3, 3>(3, 0) += HRt.transpose();
            Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;
            H6.block<6, 6>(6 * i, 6 * i) += Hb;
        }
        for (int i = 0; i < W - 1; ++i) {
            if (f.win[i].N == 0) continue;
            const double ni = f.win[i].N;
            for (int j = i + 1; j < W; ++j) {
                if (f.win[j].N == 0) continue;
                const double nj = f.win[j].N;
                Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
                Hb.block<3, 3>(0, 0) += -2.0 / NN / NN * viRiTuk[i] * viRiTuk[j].transpose();
                Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukT[i];
                Hb.block<3, 3>(3, 0) += -2.0 * ni / NN / NN * viRiTukukT[j].transpose();
                Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukT;
                H6.block<6, 6>(6 * i, 6 * j) += Hb;
            }
        }
    }
    if (hess) {
        for (int i = 0; i < W; ++i) {
            jact->segment<6>(kDim * i) += g6.segment<6>(6 * i);
            for (int j = 0; j < W; ++j) {
                // only the upper triangle was filled
                if (j >= i) {
                    hess->block<6, 6>(kDim * i, kDim * j) += H6.block<6, 6>(6 * i, 6 * j);
                } else {
                    hess->block<6, 6>(kDim * i, kDim * j) += H6.block<6, 6>(6 * j, 6 * i).transpose();
                }
            }
        }
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
    ImuPreint::Mat18 jtj;
    ImuPreint::Vec18 gg;
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

// Voxel-SLAM LI_BA_Optimizer::damping_iter: Levenberg-Marquardt with the oldest pose held fixed.
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
            if (iter == 0) residual0 = residual1;
            // the gauge is the oldest pose; its velocity stays free
            Hess.topRows(6).setZero();
            Hess.leftCols(6).setZero();
            Hess.block<6, 6>(0, 0).setIdentity();
            JacT.head(6).setZero();
        }
        D.diagonal() = Hess.diagonal();
        dxi = (Hess + u * D).ldlt().solve(-JacT);
        for (int j = 0; j < W; ++j) {
            xs_temp[j].R = xs[j].R * ExpMat(dxi.segment<3>(kDim * j));
            xs_temp[j].p = xs[j].p + dxi.segment<3>(kDim * j + 3);
            xs_temp[j].v = xs[j].v + dxi.segment<3>(kDim * j + 6);
            xs_temp[j].bg = xs[j].bg;
            xs_temp[j].ba = xs[j].ba;
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
              << std::min(residual1, residual2) << " moved " << moved << " voxels " << map_.size();
}

Trajectory LocalBA::CorrectTrajectory(const Trajectory& traj, double time_offset) const {
    Trajectory out = traj;
    if (history_.empty()) return out;
    size_t k = 0;
    for (StampedPose& sp : out) {
        const int64_t t_ns = S_TO_NS(sp.time - time_offset);
        while (k + 1 < history_.size() &&
               std::abs(history_[k + 1].time_ns - t_ns) <= std::abs(history_[k].time_ns - t_ns)) {
            ++k;
        }
        sp.pose = history_[k].refined * history_[k].odom.inverse() * sp.pose;
    }
    return out;
}

}  // namespace faster_lio
