#include "fast_lio.h"

#include <Eigen/Eigenvalues>
#include <unordered_map>

#include "timer.h"

namespace faster_lio {
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

std::vector<Vec3> FastLio::UniformSample(const std::vector<Vec3>& scan, double size) const {
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
    std::vector<Vec3> out;
    out.reserve(best.size());
    for (const Best& b : best) out.push_back(scan[b.idx]);
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

void FastLio::ProcessMeasurement(MeasureGroup& measures) {
    Timer timer;
    for (const Imu& imu : measures.imu_) lba->AddImu(imu.time_ns, imu.angular_velocity, imu.linear_acceleration);

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
              << measures.lidar_->size() << " degrade " << degrade_cnt_ << " bg " << x.bg.transpose() << " ba "
              << x.ba.transpose();
}

}  // namespace faster_lio
