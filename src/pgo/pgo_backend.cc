#include "pgo/pgo_backend.h"

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <iomanip>

#include "pgo/plane_cloud.h"
#include "so3_math.h"
#include "timer.h"

namespace faster_lio {
namespace pgo {

PgoBackend::PgoBackend(const PgoParam& prm) : prm_(prm), graph_(prm), detector_(prm) {}

void PgoBackend::AddScan(double time, const Eigen::Isometry3d& raw_pose, const std::vector<Vec3>& body_pts) {
    const Eigen::Isometry3d pose = dx_ * raw_pose;
    const bool first = scan_nodes_.empty();
    const Eigen::Isometry3d prev_pose = first ? pose : scan_nodes_.back().pose;
    if (!first) jour_ += (pose.translation() - prev_pose.translation()).norm();

    ScanNode node;
    node.time = time;
    node.pose = pose;
    scan_nodes_.push_back(node);
    init_poses_.push_back(raw_pose);
    Vec6 v6;
    v6 << Vec3::Constant(prm_.odom_var_rot), Vec3::Constant(prm_.odom_var_trans);
    graph_.AddNode(static_cast<int>(scan_nodes_.size()) - 1, pose, prev_pose, v6);

    buf_scans_.push_back(body_pts);

    if (static_cast<int>(buf_scans_.size()) < prm_.kf_scans) return;
    SealKeyframe(pose);
}

void PgoBackend::SealKeyframe(const Eigen::Isometry3d& anchor_pose) {
    if (!kfs_.empty()) {
        const Eigen::Isometry3d d = kfs_.back().pose.inverse() * anchor_pose;
        const double ang = LogMat(Mat3(d.linear())).norm() * 57.2957795;
        const double len = d.translation().norm();
        if (ang < prm_.kf_min_angle && len < prm_.kf_min_dist) {
            // standing still: let the window slide on rather than pile up near-identical keyframes
            buf_scans_.pop_front();
            return;
        }
    }

    std::vector<Vec3> merged;
    const double r2 = prm_.kf_max_range * prm_.kf_max_range;
    const size_t first_node = scan_nodes_.size() - buf_scans_.size();
    for (size_t i = 0; i < buf_scans_.size(); ++i) {
        const Eigen::Isometry3d d = anchor_pose.inverse() * scan_nodes_[first_node + i].pose;
        for (const Vec3& p : buf_scans_[i]) {
            const Vec3 q = d * p;
            if (prm_.kf_max_range > 0.0 && q.squaredNorm() > r2) continue;
            merged.push_back(q);
        }
    }
    buf_scans_.clear();

    KeyFrame kf;
    kf.id = static_cast<int>(kfs_.size());
    kf.anchor = static_cast<int>(scan_nodes_.size()) - 1;
    kf.jour = jour_;
    kf.pose = anchor_pose;
    kf.cloud = DownSample(merged, prm_.kf_ds_voxel);
    kf.planes = ExtractPlanes(kf.cloud, prm_);
    kfs_.push_back(kf);
    ++kf_since_opt_;

    LoopEdge edge;
    if (!detector_.Detect(kfs_, kf.id, &edge)) return;

    graph_.AddLoop(kfs_[edge.from].anchor, kfs_[edge.to].anchor, edge.T_from_to);
    loops_.push_back(edge);
    LOG(INFO) << "PGO: loop kf " << edge.from << " -> " << edge.to << " drift " << edge.drift
              << " m over " << edge.span << " m, lambda_min " << edge.eigval;

    // the paper only pays for a solve and a map rebuild once the drift is worth it
    unsolved_drift_ = std::max(unsolved_drift_, edge.drift);
    if (kf_since_opt_ > prm_.opt_halt_kf && edge.drift > prm_.opt_min_drift)
        Solve();
}

void PgoBackend::Finish() {
    if (unsolved_drift_ > prm_.opt_min_drift) Solve();
}

void PgoBackend::Solve() {
    Timer timer;
    Isometries poses(scan_nodes_.size());
    for (size_t i = 0; i < scan_nodes_.size(); ++i) poses[i] = scan_nodes_[i].pose;
    const Eigen::Isometry3d old_last = scan_nodes_.back().pose;

    if (!graph_.Solve(&poses)) return;

    for (size_t i = 0; i < scan_nodes_.size(); ++i) scan_nodes_[i].pose = poses[i];
    dx_ = scan_nodes_.back().pose * init_poses_.back().inverse();
    for (KeyFrame& kf : kfs_) kf.pose = scan_nodes_[kf.anchor].pose;

    kf_since_opt_ = 0;
    unsolved_drift_ = 0.0;
    ++num_solves_;
    const Eigen::Isometry3d step = scan_nodes_.back().pose * old_last.inverse();
    LOG(INFO) << "PGO: solved " << scan_nodes_.size() << " nodes, " << graph_.num_loops() << " loops, dx t "
              << step.translation().norm() << " m r "
              << LogMat(Mat3(step.linear())).norm() * 57.2957795 << " deg in " << timer.ElapsedMiniSeconds()
              << " ms";
}

void PgoBackend::SaveDebug(const std::string& output_dir) const {
    {
        std::ofstream f(output_dir + "/pgo_keyframes.txt");
        f << "# id anchor jour t x y z qx qy qz qw points planes\n";
        for (const KeyFrame& kf : kfs_) {
            const Eigen::Quaterniond q(kf.pose.linear());
            f << kf.id << " " << kf.anchor << " " << kf.jour << " " << std::fixed << std::setprecision(9)
              << scan_nodes_[kf.anchor].time << " " << kf.pose.translation().transpose() << " " << q.x() << " "
              << q.y() << " " << q.z() << " " << q.w() << " " << kf.cloud.size() << " " << kf.planes.size()
              << "\n";
            f << std::defaultfloat;
        }
    }
    {
        std::ofstream f(output_dir + "/pgo_loops.txt");
        f << "# from to drift span ratio lambda_min\n";
        for (const LoopEdge& e : loops_) {
            f << e.from << " " << e.to << " " << e.drift << " " << e.span << " " << e.drift / e.span << " "
              << e.eigval << "\n";
        }
    }
    {
        std::ofstream f(output_dir + "/pgo_rejections.txt");
        f << "# from to reason value matches iters step nn_median src_n tar_n seed_t\n";
        for (const LoopRejection& r : detector_.rejections()) {
            f << r.from << " " << r.to << " " << r.reason << " " << r.value << " " << r.matches << " "
              << r.iters << " " << r.step << " " << r.nn_median << " " << r.src_n << " " << r.tar_n << " "
              << r.seed_t << "\n";
        }
    }
    LOG(INFO) << "PGO: " << scan_nodes_.size() << " nodes, " << kfs_.size() << " keyframes, " << loops_.size()
              << " loops, " << detector_.rejections().size() << " rejected candidates, " << num_solves_
              << " solves";
}

}  // namespace pgo
}  // namespace faster_lio
