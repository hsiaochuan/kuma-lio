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

namespace kuma_lio {
namespace pgo {

PgoBackend::PgoBackend(const PgoParam& prm) : prm_(prm), graph_(prm), detector_(prm) {}

void PgoBackend::AddScan(double time, const Eigen::Isometry3d& init_pose, const std::vector<Vec3>& scan_points) {
    const Eigen::Isometry3d cor_pose = dx_ * init_pose;
    const bool if_first = scan_nodes_.empty();
    const Eigen::Isometry3d prev_pose = if_first ? cor_pose : scan_nodes_.back().pose;
    if (!if_first) travel_len_ += (cor_pose.translation() - prev_pose.translation()).norm();

    ScanNode scan_node;
    scan_node.time = time;
    scan_node.pose = cor_pose;
    scan_nodes_.push_back(scan_node);
    init_poses_.push_back(init_pose);
    Vec6 v6;
    v6 << Vec3::Constant(prm_.odom_var_rot), Vec3::Constant(prm_.odom_var_trans);
    graph_.AddNode(static_cast<int>(scan_nodes_.size()) - 1, cor_pose, prev_pose, v6);

    buf_scans_.push_back(scan_points);

    if (static_cast<int>(buf_scans_.size()) < prm_.kf_scans) return;
    if (!kfs_.empty()) {
        const Eigen::Isometry3d rel = kfs_.back().pose.inverse() * cor_pose;
        const double rel_angle = LogMat(Mat3(rel.linear())).norm() * 180.0 / PI;
        const double len = rel.translation().norm();
        if (rel_angle < prm_.kf_min_angle && len < prm_.kf_min_dist) {
            buf_scans_.pop_front();
            return;
        }
    }

    std::vector<Vec3> kf_points;
    PlaneCells cells;
    SealKeyframe(cor_pose, &kf_points, &cells);
    ++kf_since_opt_;

    LoopEdge loop;
    if (!detector_.Detect(kfs_, kf_points, cells, &loop)) return;
    graph_.AddLoop(kfs_[loop.from].scan_id, kfs_[loop.to].scan_id, loop.T_from_to);
    loops_.push_back(loop);
    LOG(INFO) << "PGO: " << LoopSourceName(loop.source) << " loop kf " << loop.from << " -> " << loop.to << " drift "
              << loop.drift << " m over " << loop.span << " m, lambda_min " << loop.eigval;

    // the paper only pays for a solve and a map rebuild once the drift is worth it
    ++unsolved_loops_;
    if (kf_since_opt_ > prm_.opt_cooldown_kf && loop.drift > prm_.opt_min_drift) Solve();
}

void PgoBackend::SealKeyframe(const Eigen::Isometry3d& scan_pose, std::vector<Vec3>* kf_points, PlaneCells* cells) {
    const size_t first_buf_scan = scan_nodes_.size() - buf_scans_.size();
    for (size_t i = 0; i < buf_scans_.size(); ++i) {
        const Eigen::Isometry3d rel = scan_pose.inverse() * scan_nodes_[first_buf_scan + i].pose;
        for (const Vec3& p : buf_scans_[i]) {
            const Vec3 q = rel * p;
            kf_points->push_back(q);
        }
    }
    buf_scans_.clear();

    KeyFrame kf;
    kf.kf_id = static_cast<int>(kfs_.size());
    kf.scan_id = static_cast<int>(scan_nodes_.size()) - 1;
    kf.travel_len = travel_len_;
    kf.pose = scan_pose;
    const std::vector<Vec3> cloud = UniformSample(*kf_points, prm_.kf_ds_voxel);
    *cells = ExtractPlaneCells(cloud, prm_);
    kf.points = static_cast<int>(cloud.size());
    kf.planes = PlanePoints(*cells);
    kfs_.push_back(kf);
}

void PgoBackend::Finish() {
    // every loop still waiting, however small its drift: on building00 this tail decided 0.19 vs 0.41 m ATE
    if (unsolved_loops_ > 0) Solve();
}

void PgoBackend::Solve() {
    Timer timer;
    Isometries poses(scan_nodes_.size());
    for (size_t i = 0; i < scan_nodes_.size(); ++i) poses[i] = scan_nodes_[i].pose;
    const Eigen::Isometry3d old_last = scan_nodes_.back().pose;

    if (!graph_.Solve(&poses)) return;

    for (size_t i = 0; i < scan_nodes_.size(); ++i) scan_nodes_[i].pose = poses[i];
    dx_ = scan_nodes_.back().pose * init_poses_.back().inverse();
    for (KeyFrame& kf : kfs_) kf.pose = scan_nodes_[kf.scan_id].pose;

    kf_since_opt_ = 0;
    unsolved_loops_ = 0;
    ++num_solves_;
    const Eigen::Isometry3d step = scan_nodes_.back().pose * old_last.inverse();
    LOG(INFO) << "PGO: solved " << scan_nodes_.size() << " nodes, " << graph_.num_loops() << " loops, dx t "
              << step.translation().norm() << " m r " << LogMat(Mat3(step.linear())).norm() * 180.0 / PI << " deg in "
              << timer.ElapsedMiniSeconds() << " ms";
}

void PgoBackend::SaveDebug(const std::string& output_dir) const {
    {
        std::ofstream f(output_dir + "/pgo_keyframes.txt");
        f << "# kf_id scan_id travel_len t x y z qx qy qz qw points planes btc_keypoints btc_triangles\n";
        for (const KeyFrame& kf : kfs_) {
            const Eigen::Quaterniond q(kf.pose.linear());
            const BtcFrame* btc = detector_.btc_frame(kf.kf_id);
            f << kf.kf_id << " " << kf.scan_id << " " << kf.travel_len << " " << std::fixed << std::setprecision(9)
              << scan_nodes_[kf.scan_id].time << " " << kf.pose.translation().transpose() << " " << q.x() << " "
              << q.y() << " " << q.z() << " " << q.w() << " " << kf.points << " " << kf.planes.size() << " "
              << (btc ? btc->keypoints.size() : 0) << " " << (btc ? btc->triangles.size() : 0) << "\n";
            f << std::defaultfloat;
        }
    }
    {
        std::ofstream f(output_dir + "/pgo_loops.txt");
        f << "# from to drift span ratio lambda_min tx ty tz qx qy qz qw (T_from_to) source\n";
        for (const LoopEdge& e : loops_) {
            const Eigen::Quaterniond q(e.T_from_to.linear());
            f << e.from << " " << e.to << " " << e.drift << " " << e.span << " " << e.drift / e.span << " " << e.eigval
              << " " << e.T_from_to.translation().transpose() << " " << q.x() << " " << q.y() << " " << q.z() << " "
              << q.w() << " " << LoopSourceName(e.source) << "\n";
        }
    }
    {
        std::ofstream f(output_dir + "/pgo_rejections.txt");
        f << "# from to reason value matches iters step nn_median src_n tar_n seed_t source\n";
        for (const LoopRejection& r : detector_.rejections()) {
            f << r.from << " " << r.to << " " << r.reason << " " << r.value << " " << r.matches << " " << r.iters << " "
              << r.step << " " << r.nn_median << " " << r.src_n << " " << r.tar_n << " " << r.seed_t << " "
              << LoopSourceName(r.source) << "\n";
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
            const Eigen::Vector3d a = scan_nodes_[kfs_[e.from].scan_id].pose.translation();
            const Eigen::Vector3d b = scan_nodes_[kfs_[e.to].scan_id].pose.translation();
            const int n = std::max(1, static_cast<int>(std::ceil((b - a).norm() / kLineStep)));
            for (int k = 1; k < n; ++k) add(a + (b - a) * (static_cast<double>(k) / n), 255, 0, 0);
        }
        pcl::io::savePLYFileBinary(output_dir + "/pgo_graph.ply", cloud);
    }
    LOG(INFO) << "PGO: " << scan_nodes_.size() << " nodes, " << kfs_.size() << " keyframes, " << loops_.size()
              << " loops, " << detector_.rejections().size() << " rejected candidates, " << num_solves_ << " solves";
}

}  // namespace pgo
}  // namespace kuma_lio
