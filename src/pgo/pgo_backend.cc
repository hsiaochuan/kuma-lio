#include "pgo/pgo_backend.h"

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "pgo/loop_detector.h"
#include "pgo/pose_graph.h"

namespace faster_lio {
namespace pgo {
namespace {

double RotationAngle(const Eigen::Matrix3d& R) { return Eigen::AngleAxisd(R).angle(); }

enum class Which { kOdom, kOpt };

void WriteTum(const std::string& fname, const std::vector<KeyFrame::Ptr>& kfs, Which which) {
    std::ofstream ofs(fname);
    if (!ofs.is_open()) {
        LOG(WARNING) << "[pgo] cannot write " << fname;
        return;
    }
    ofs << std::fixed << std::setprecision(9);
    for (const KeyFrame::Ptr& kf : kfs) {
        const Eigen::Isometry3d& T = which == Which::kOpt ? kf->pose_opt : kf->pose_odom;
        Eigen::Quaterniond q(T.linear());
        std::ostringstream ss;
        ss << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << kf->time;
        ofs << ss.str() << " " << T.translation().x() << " " << T.translation().y() << " "
            << T.translation().z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
            << "\n";
    }
}

}  // namespace

PgoBackend::PgoBackend(const PgoParam& prm) : prm_(prm), staging_points_(new PgoCloud) {
    graph_.reset(new PoseGraph(prm_));
    loop_.reset(new LoopDetector(prm_));
    LOG(INFO) << "[pgo] offline batch: keyframes are only stored during the run; the graph, "
                 "the loop closures and the optimization all happen in Finalize()";
}

// Out of line: PoseGraph and LoopDetector are incomplete in the header.
PgoBackend::~PgoBackend() = default;

void PgoBackend::AddKnot(double t_abs, const Eigen::Isometry3d& pose_w_b,
                         const PgoCloud& world_pts) {

    if (!std::isfinite(t_abs)) return;

    if (!have_anchor_) {
        last_kf_pose_odom_ = pose_w_b;
        last_kf_time_ = t_abs;
        have_anchor_ = true;
        staging_points_->clear();
    }
    cur_pose_odom_ = pose_w_b;
    cur_time_ = t_abs;

    // Safety valve: with kf_max_dt disabled a stationary platform would grow
    // the staging buffer without bound.  Dropping points once it is already
    // over-full is O(1); the keyframe has more than enough by then.
    if (!world_pts.empty() &&
        static_cast<int>(staging_points_->size()) < prm_.kf_max_staging_points) {
        staging_points_->reserve(staging_points_->size() + world_pts.size());
        staging_points_->insert(staging_points_->end(), world_pts.begin(), world_pts.end());
    }

    const Eigen::Isometry3d d = last_kf_pose_odom_.inverse() * pose_w_b;
    const double dist = d.translation().norm();
    const double ang = RotationAngle(d.linear());
    const double dt = t_abs - last_kf_time_;

    const bool seal = dist > prm_.kf_dis_th || ang > prm_.kf_angle_th_deg * M_PI / 180.0 ||
                      (prm_.kf_max_dt > 0.0 && dt > prm_.kf_max_dt);
    if (!seal) return;

    if (static_cast<int>(staging_points_->size()) < prm_.kf_min_points) {
        LOG(INFO) << "Too few points in keyframe, drop it";
        last_kf_pose_odom_ = pose_w_b;
        last_kf_time_ = t_abs;
        return;
    }
    SealKeyFrame(t_abs, pose_w_b);
}

void PgoBackend::SealKeyFrame(double t_abs, const Eigen::Isometry3d& pose_w_b) {
    KeyFrame::Ptr kf(new KeyFrame);
    kf->id = static_cast<int>(kfs_.size());
    kf->time = t_abs;
    kf->pose_odom = pose_w_b;
    kf->pose_opt = pose_w_b;  // until Finalize() says otherwise
    kf->cloud = staging_points_;
    staging_points_.reset(new PgoCloud);

    last_kf_pose_odom_ = pose_w_b;
    last_kf_time_ = t_abs;

    // ---- 1. range crop, in world coordinates ----
    if (prm_.kf_max_range > 0.0 && !kf->cloud->empty()) {
        const double r2 = prm_.kf_max_range * prm_.kf_max_range;
        const Eigen::Vector3f c = kf->pose_odom.translation().cast<float>();
        PgoCloudPtr cropped(new PgoCloud);
        cropped->reserve(kf->cloud->size());
        for (const PgoPoint& p : kf->cloud->points) {
            if (!std::isfinite(p.x)) continue;
            if ((p.getVector3fMap() - c).squaredNorm() <= r2) cropped->push_back(p);
        }
        kf->cloud = cropped;
    }

    // ---- 2. downsample (bounded extent, so VoxelGrid cannot overflow) ----
    if (prm_.kf_voxel_leaf > 0.0 && !kf->cloud->empty()) {
        PgoCloudPtr ds(new PgoCloud);
        pcl::VoxelGrid<PgoPoint> vg;
        const float leaf = static_cast<float>(prm_.kf_voxel_leaf);
        vg.setLeafSize(leaf, leaf, leaf);
        vg.setInputCloud(kf->cloud);
        vg.filter(*ds);
        kf->cloud = ds;
    }

    // ---- 3. store in the keyframe body frame ----
    if (!kf->cloud->empty()) {
        pcl::transformPointCloud(*kf->cloud, *kf->cloud,
                                 kf->pose_odom.inverse().matrix().cast<float>());
    }
    kf->cloud->width = kf->cloud->size();
    kf->cloud->height = 1;

    // That is all that happens while the bag is being played: no graph, no loop.
    kfs_.push_back(kf);
}

void PgoBackend::BuildGraph() {
    // A prior plus the odometry chain.  Its solution is pose_odom itself, which
    // is also the initial value, so nothing needs solving until a loop arrives.
    graph_->AddPrior(0, kfs_[0]->pose_odom);
    for (size_t i = 1; i < kfs_.size(); ++i) {
        const Eigen::Isometry3d rel = kfs_[i - 1]->pose_odom.inverse() * kfs_[i]->pose_odom;
        graph_->AddOdometry(static_cast<int>(i) - 1, static_cast<int>(i), rel, kfs_[i]->pose_odom);
    }
}

void PgoBackend::RefreshOptimizedPoses() {
    for (const KeyFrame::Ptr& kf : kfs_) kf->pose_opt = graph_->Pose(kf->id);
}

void PgoBackend::DetectLoops() {
    if (!prm_.loop_enable || kfs_.size() < 2) return;
    LOG(INFO) << "[pgo] loop sweep over " << kfs_.size() << " keyframes";
    for (size_t i = 0; i < kfs_.size(); ++i) {
        if (static_cast<int>(loops_.size()) >= prm_.final_sweep_max_loops) break;
        LoopEdge e;
        if (!loop_->Detect(kfs_, static_cast<int>(i), &e) || !LoopIsInformative(e)) continue;
        graph_->AddLoop(e);
        loops_.push_back(e);
        // Re-solve straight away: the candidate search and the submaps of every
        // later keyframe use pose_opt, so each loop makes the next one easier
        // to find.
        graph_->Optimize();
        RefreshOptimizedPoses();
    }
    LOG(INFO) << "[pgo] loop sweep added " << loops_.size() << " loops";
}

void PgoBackend::Finalize() {
    // Seal what is still accumulating as one last keyframe -- unless the
    // platform has barely moved, which would only add a zero-motion edge.
    if (have_anchor_ && static_cast<int>(staging_points_->size()) >= prm_.kf_min_points &&
        (last_kf_pose_odom_.inverse() * cur_pose_odom_).translation().norm() >
            0.2 * prm_.kf_dis_th) {
        SealKeyFrame(cur_time_, cur_pose_odom_);
    }

    if (kfs_.empty()) {
        LOG(WARNING) << "[pgo] no keyframes were built";
        return;
    }

    try {
        BuildGraph();
        DetectLoops();
        if (loops_.empty()) {
            LOG(INFO) << "[pgo] " << kfs_.size() << " keyframes, no loops -- nothing to correct";
            return;
        }
        // DetectLoops() left the estimate solved for every loop it accepted.
        if (prm_.final_outlier_reject) {
            for (int r = 0; r < std::max(1, prm_.final_reject_rounds); ++r) {
                const int dropped = graph_->RejectLoopOutliers();
                if (dropped == 0) break;
                LOG(INFO) << "[pgo] dropped " << dropped << " loop factors with chi2 > "
                          << prm_.final_outlier_chi2;
                graph_->Optimize();
            }
        }
        RefreshOptimizedPoses();
    } catch (const std::exception& e) {
        // A sweep that died half way has already moved some keyframes; put them
        // all back on odometry so CorrectTrajectory() becomes the identity.
        LOG(ERROR) << "[pgo] optimization failed: " << e.what()
                   << " -- falling back to the uncorrected trajectory";
        for (const KeyFrame::Ptr& kf : kfs_) kf->pose_opt = kf->pose_odom;
        return;
    }

    double max_shift = 0.0;
    for (const KeyFrame::Ptr& kf : kfs_) {
        max_shift = std::max(max_shift,
                             (kf->pose_opt.translation() - kf->pose_odom.translation()).norm());
    }
    LOG(INFO) << "[pgo] " << kfs_.size() << " keyframes, " << loops_.size()
              << " loops, graph error " << graph_->Error() << ", max keyframe shift " << max_shift
              << " m";
}

bool PgoBackend::LoopIsInformative(const LoopEdge& e) const {
    if (prm_.loop_min_odom_sigma_ratio <= 0.0) return true;
    const int span = std::abs(e.id_to - e.id_from);
    if (span <= 0) return false;
    // Random walk over `span` odometry edges, each with sigma odom_sigma_trans.
    const double odom_sigma = prm_.odom_sigma_trans * std::sqrt(static_cast<double>(span));
    const double need = prm_.loop_min_odom_sigma_ratio * prm_.loop_sigma_trans;
    if (odom_sigma > need) return true;
    LOG(INFO) << "[pgo] loop " << e.id_from << " -> " << e.id_to << " carries no information: "
              << "odometry sigma over " << span << " keyframes is " << odom_sigma
              << " m, below " << need << " m of loop sigma";
    return false;
}

Trajectory PgoBackend::CorrectTrajectory(const Trajectory& traj_odom) const {
    if (kfs_.size() < 2 || traj_odom.empty()) return traj_odom;

    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> corr(kfs_.size());
    for (size_t k = 0; k < kfs_.size(); ++k) {
        // The correction lives in the WORLD frame, hence a LEFT multiply below.
        // A right multiply would be a body-frame transform and would rotate the
        // trajectory about itself instead of relocating it.
        corr[k] = kfs_[k]->pose_opt * kfs_[k]->pose_odom.inverse();
    }

    Trajectory out;
    out.reserve(traj_odom.size());
    size_t k = 0;
    for (const StampedPose& sp : traj_odom) {
        while (k + 2 < kfs_.size() && kfs_[k + 1]->time <= sp.time) ++k;

        Eigen::Isometry3d Tc;
        if (sp.time <= kfs_.front()->time) {
            Tc = corr.front();
        } else if (sp.time >= kfs_.back()->time) {
            Tc = corr.back();
        } else {
            const double t0 = kfs_[k]->time;
            const double t1 = kfs_[k + 1]->time;
            double a = (t1 > t0) ? (sp.time - t0) / (t1 - t0) : 0.0;
            a = std::min(std::max(a, 0.0), 1.0);
            // Interpolating the correction is exactly a smooth blend between
            // chaining forward from keyframe k and backward from k+1.
            Tc = interpolate_pose(corr[k], corr[k + 1], a);
        }
        out.emplace_back(sp.time, Eigen::Isometry3d(Tc * sp.pose));
    }
    return out;
}

void PgoBackend::SaveDebug(const std::string& output_dir) const {
    if (kfs_.empty()) return;
    WriteTum(output_dir + "/pgo_kf_odom.txt", kfs_, Which::kOdom);
    WriteTum(output_dir + "/pgo_kf_opt.txt", kfs_, Which::kOpt);

    std::ofstream ofs(output_dir + "/pgo_loops.txt");
    if (!ofs.is_open()) return;
    ofs << "# loops: id_from id_to fitness overlap corr_trans corr_rot_deg chi2 kept\n";
    const std::vector<double> chi2 = graph_->LoopChi2();  // -1 marks a rejected loop
    for (size_t i = 0; i < loops_.size(); ++i) {
        const LoopEdge& e = loops_[i];
        ofs << e.id_from << " " << e.id_to << " " << e.fitness << " " << e.overlap << " "
            << e.corr_trans << " " << e.corr_rot << " " << chi2[i] << " " << int(chi2[i] >= 0.0)
            << "\n";
    }
    ofs << "# rejected: id_from id_to reason fitness overlap corr_trans corr_rot_deg\n";
    for (const LoopDetector::Rejection& r : loop_->rejections()) {
        ofs << r.id_from << " " << r.id_to << " " << r.reason << " " << r.fitness << " "
            << r.overlap << " " << r.corr_trans << " " << r.corr_rot << "\n";
    }
}

}  // namespace pgo
}  // namespace faster_lio
