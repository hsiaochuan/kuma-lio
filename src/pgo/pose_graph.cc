#include "pgo/pose_graph.h"

#include <glog/logging.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

namespace faster_lio {
namespace pgo {
namespace {

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

struct PoseGraph::Impl {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
};

PoseGraph::PoseGraph(const PgoParam& prm) : impl_(new Impl), prm_(prm) {}
PoseGraph::~PoseGraph() = default;

void PoseGraph::AddNode(int index, const Eigen::Isometry3d& pose, const Eigen::Isometry3d& prev_pose,
                        const Vec6& prev_v6) {
    impl_->initial.insert(index, ToPose3(pose));
    if (index == 0) {
        impl_->graph.addPrior(0, ToPose3(pose),
                              gtsam::noiseModel::Diagonal::Variances(Vec6::Constant(prm_.prior_var)));
        return;
    }
    // gtsam::Pose3 orders its tangent space ROTATION FIRST: [rx ry rz tx ty tz], and so does v6.
    const Eigen::Isometry3d T_prev_cur = prev_pose.inverse() * pose;
    impl_->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        index - 1, index, ToPose3(T_prev_cur), gtsam::noiseModel::Diagonal::Variances(prev_v6));
}

void PoseGraph::AddLoop(int from, int to, const Eigen::Isometry3d& T_from_to) {
    // Voxel-SLAM uses no robust kernel here: everything rests on the gating in LoopDetector.
    impl_->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        from, to, ToPose3(T_from_to),
        gtsam::noiseModel::Diagonal::Variances(Vec6::Constant(prm_.loop_var)));
    ++num_loops_;
}

bool PoseGraph::Solve(Isometries* poses) {
    if (impl_->initial.empty()) return false;
    // A fresh ISAM2 fed the whole graph, as in voxelslam.cpp: batch Gauss-Newton in practice, but
    // the graph is small enough at a 10 Hz node rate that incremental bookkeeping buys nothing.
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    gtsam::ISAM2 isam(params);
    gtsam::Values results;
    try {
        isam.update(impl_->graph, impl_->initial);
        for (int i = 0; i < prm_.isam_extra_iter; ++i) isam.update();
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
    impl_->initial = results;
    return true;
}

}  // namespace pgo
}  // namespace faster_lio
