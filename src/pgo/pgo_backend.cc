#include "pgo/pgo_backend.h"

#include <glog/logging.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
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

    ScanNode scan_node;
    scan_node.time = time;
    scan_node.pose = pose;
    scan_nodes_.push_back(scan_node);
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
    const std::vector<Vec3> cloud = UniformSample(merged, prm_.kf_ds_voxel);
    const PlaneCells cells = ExtractPlaneCells(cloud, prm_);
    kf.points = static_cast<int>(cloud.size());
    kf.planes = PlanePoints(cells);
    kfs_.push_back(kf);
    ++kf_since_opt_;

    LoopEdge loop;
    if (!detector_.Detect(kfs_, merged, cells, &loop)) return;

    graph_.AddLoop(kfs_[loop.from].anchor, kfs_[loop.to].anchor, loop.T_from_to);
    loops_.push_back(loop);
    LOG(INFO) << "PGO: loop kf " << loop.from << " -> " << loop.to << " drift " << loop.drift
              << " m over " << loop.span << " m, lambda_min " << loop.eigval;

    // the paper only pays for a solve and a map rebuild once the drift is worth it
    unsolved_drift_ = std::max(unsolved_drift_, loop.drift);
    ++unsolved_loops_;
    if (kf_since_opt_ > prm_.opt_halt_kf && loop.drift > prm_.opt_min_drift)
        Solve();
}

void PgoBackend::Finish() {
    if (unsolved_drift_ > prm_.opt_min_drift || (prm_.opt_final_all && unsolved_loops_ > 0)) Solve();
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
    unsolved_loops_ = 0;
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
        f << "# id anchor jour t x y z qx qy qz qw points planes btc_keypoints btc_triangles\n";
        for (const KeyFrame& kf : kfs_) {
            const Eigen::Quaterniond q(kf.pose.linear());
            const BtcFrame* btc = detector_.btc_frame(kf.id);
            f << kf.id << " " << kf.anchor << " " << kf.jour << " " << std::fixed << std::setprecision(9)
              << scan_nodes_[kf.anchor].time << " " << kf.pose.translation().transpose() << " " << q.x() << " "
              << q.y() << " " << q.z() << " " << q.w() << " " << kf.points << " " << kf.planes.size() << " "
              << (btc ? btc->keypoints.size() : 0) << " " << (btc ? btc->triangles.size() : 0) << "\n";
            f << std::defaultfloat;
        }
    }
    {
        std::ofstream f(output_dir + "/pgo_loops.txt");
        f << "# from to drift span ratio lambda_min tx ty tz qx qy qz qw (T_from_to)\n";
        for (const LoopEdge& e : loops_) {
            const Eigen::Quaterniond q(e.T_from_to.linear());
            f << e.from << " " << e.to << " " << e.drift << " " << e.span << " " << e.drift / e.span << " "
              << e.eigval << " " << e.T_from_to.translation().transpose() << " " << q.x() << " " << q.y() << " "
              << q.z() << " " << q.w() << "\n";
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
    {
        // the optimized graph as a point cloud: every node a white point, every loop a red line between
        // its two anchors, sampled as points because viewers do not draw PLY edges
        constexpr double kLineStep = 0.02;  // [m]
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        auto add = [&cloud](const Eigen::Vector3d& p, uint8_t r, uint8_t g, uint8_t b) {
            pcl::PointXYZRGB q(r, g, b);
            q.x = p.x();
            q.y = p.y();
            q.z = p.z();
            cloud.push_back(q);
        };
        for (const ScanNode& n : scan_nodes_) add(n.pose.translation(), 255, 255, 255);
        for (const LoopEdge& e : loops_) {
            const Eigen::Vector3d a = scan_nodes_[kfs_[e.from].anchor].pose.translation();
            const Eigen::Vector3d b = scan_nodes_[kfs_[e.to].anchor].pose.translation();
            const int n = std::max(1, static_cast<int>(std::ceil((b - a).norm() / kLineStep)));
            for (int k = 1; k < n; ++k) add(a + (b - a) * (static_cast<double>(k) / n), 255, 0, 0);
        }
        pcl::io::savePLYFileBinary(output_dir + "/pgo_graph.ply", cloud);
    }
    LOG(INFO) << "PGO: " << scan_nodes_.size() << " nodes, " << kfs_.size() << " keyframes, " << loops_.size()
              << " loops, " << detector_.rejections().size() << " rejected candidates, " << num_solves_
              << " solves";
}

}  // namespace pgo
}  // namespace faster_lio
