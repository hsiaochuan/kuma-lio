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

#include "pgo/pose_graph.h"

#include <glog/logging.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>

namespace kuma_lio {
namespace pgo {
namespace {

constexpr int kIsamExtraIters = 5;  // extra ISAM2 updates after the first, to converge the relinearization

inline gtsam::Pose3 ToPose3(const Eigen::Isometry3d& T) {
    return gtsam::Pose3(gtsam::Rot3(T.linear()), gtsam::Point3(T.translation()));
}

inline Eigen::Isometry3d FromPose3(const gtsam::Pose3& p) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = p.rotation().matrix();
    T.translation() = p.translation();
    return T;
}

}  // namespace

void PoseGraph::AddNode(int index, const Eigen::Isometry3d& pose, const Eigen::Isometry3d& prev_pose,
                        const Vec6& prev_v6) {
    initial_.insert(index, ToPose3(pose));
    if (index == 0) {
        graph_.addPrior(0, ToPose3(pose), gtsam::noiseModel::Diagonal::Variances(Vec6::Constant(prm_.prior_var)));
        return;
    }
    // gtsam::Pose3 orders its tangent space ROTATION FIRST: [rx ry rz tx ty tz], and so does v6.
    const Eigen::Isometry3d T_prev_cur = prev_pose.inverse() * pose;
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        index - 1, index, ToPose3(T_prev_cur), gtsam::noiseModel::Diagonal::Variances(prev_v6));
}

void PoseGraph::AddLoop(int from, int to, const Eigen::Isometry3d& T_from_to) {
    // Voxel-SLAM uses no robust kernel here: everything rests on the gating in LoopDetector.
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        from, to, ToPose3(T_from_to), gtsam::noiseModel::Diagonal::Variances(Vec6::Constant(prm_.loop_var)));
    ++num_loops_;
}

bool PoseGraph::Solve(Isometries* poses) {
    if (initial_.empty()) return false;
    // A fresh ISAM2 fed the whole graph, as in voxelslam.cpp: batch Gauss-Newton in practice, but
    // the graph is small enough at a 10 Hz node rate that incremental bookkeeping buys nothing.
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    gtsam::ISAM2 isam(params);
    gtsam::Values results;
    try {
        isam.update(graph_, initial_);
        for (int i = 0; i < kIsamExtraIters; ++i) isam.update();
        results = isam.calculateEstimate();
    } catch (const std::exception& e) {
        LOG(WARNING) << "PGO: ISAM2 failed, keeping the odometry poses: " << e.what();
        return false;
    }
    if (results.size() != poses->size()) {
        LOG(WARNING) << "PGO: solved " << results.size() << " nodes but hold " << poses->size();
        return false;
    }
    for (size_t i = 0; i < poses->size(); ++i) {
        (*poses)[i] = FromPose3(results.at<gtsam::Pose3>(static_cast<int>(i)));
    }
    // relinearize the next solve around what we just found
    initial_ = results;
    return true;
}

}  // namespace pgo
}  // namespace kuma_lio
