#include "pgo/pose_graph.h"

#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <cmath>

namespace faster_lio {
namespace pgo {
namespace {

gtsam::Pose3 ToGtsam(const Eigen::Isometry3d& T) {
    // Build the rotation from a normalized quaternion: Rot3(Matrix3) would run
    // an orthonormalization step on every call.
    Eigen::Quaterniond q(T.linear());
    q.normalize();
    return gtsam::Pose3(gtsam::Rot3(q), gtsam::Point3(T.translation()));
}

Eigen::Isometry3d FromGtsam(const gtsam::Pose3& p) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = p.rotation().matrix();
    T.translation() = p.translation();
    return T;
}

/// gtsam::Pose3 orders its tangent space ROTATION FIRST: [rx ry rz tx ty tz].
gtsam::Vector6 Sigmas(double rot_deg, double trans_m) {
    const double r = rot_deg * M_PI / 180.0;
    gtsam::Vector6 s;
    s << r, r, r, trans_m, trans_m, trans_m;
    return s;
}

}  // namespace

struct PoseGraph::Impl {
    PgoParam prm;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values estimate;
    /// One entry per loop ever added, in order: its index into `graph`, or -1
    /// once the outlier pass has dropped it.
    std::vector<long> loop_factor_indices;

    gtsam::SharedNoiseModel prior_noise;
    gtsam::SharedNoiseModel odom_noise;
    gtsam::SharedNoiseModel loop_noise;      // robust, used in the graph
    gtsam::SharedNoiseModel loop_noise_raw;  // the underlying Gaussian, for chi2

    explicit Impl(const PgoParam& p) : prm(p) {
        prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
            Sigmas(prm.prior_sigma_rot, prm.prior_sigma_trans));
        odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
            Sigmas(prm.odom_sigma_rot, prm.odom_sigma_trans));
        loop_noise_raw = gtsam::noiseModel::Diagonal::Sigmas(
            Sigmas(prm.loop_sigma_rot, prm.loop_sigma_trans));
        loop_noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(prm.loop_robust_delta),
            gtsam::noiseModel::Diagonal::Sigmas(Sigmas(prm.loop_sigma_rot, prm.loop_sigma_trans)));
    }
};

PoseGraph::PoseGraph(const PgoParam& prm) : impl_(new Impl(prm)) {}
PoseGraph::~PoseGraph() = default;

void PoseGraph::AddPrior(int key, const Eigen::Isometry3d& pose) {
    const gtsam::Pose3 p = ToGtsam(pose);
    impl_->graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(gtsam::Key(key), p,
                                                                 impl_->prior_noise);
    impl_->estimate.insert(gtsam::Key(key), p);
}

void PoseGraph::AddOdometry(int key_prev, int key_cur, const Eigen::Isometry3d& rel,
                            const Eigen::Isometry3d& init_cur) {
    impl_->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        gtsam::Key(key_prev), gtsam::Key(key_cur), ToGtsam(rel), impl_->odom_noise);
    impl_->estimate.insert(gtsam::Key(key_cur), ToGtsam(init_cur));
}

void PoseGraph::AddLoop(const LoopEdge& edge) {
    impl_->loop_factor_indices.push_back(static_cast<long>(impl_->graph.size()));
    impl_->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        gtsam::Key(edge.id_from), gtsam::Key(edge.id_to), ToGtsam(edge.T_from_to),
        impl_->loop_noise);
}

void PoseGraph::Optimize() {
    gtsam::LevenbergMarquardtParams lp;
    lp.setMaxIterations(std::max(1, impl_->prm.final_lm_iterations));
    lp.setRelativeErrorTol(1e-5);
    lp.setAbsoluteErrorTol(1e-5);
    lp.setVerbosityLM("SILENT");
    impl_->estimate =
        gtsam::LevenbergMarquardtOptimizer(impl_->graph, impl_->estimate, lp).optimize();
}

int PoseGraph::RejectLoopOutliers() {
    const std::vector<double> chi2 = LoopChi2();
    std::vector<bool> drop(impl_->graph.size(), false);
    int dropped = 0;
    for (size_t i = 0; i < chi2.size(); ++i) {
        if (chi2[i] > impl_->prm.final_outlier_chi2) {
            drop[impl_->loop_factor_indices[i]] = true;
            ++dropped;
        }
    }
    if (dropped == 0) return 0;

    // Rebuild without the offenders rather than nulling slots: a
    // NonlinearFactorGraph with holes does not survive elimination cleanly.
    gtsam::NonlinearFactorGraph pruned;
    std::vector<long> new_index(impl_->graph.size(), -1);
    for (size_t i = 0; i < impl_->graph.size(); ++i) {
        if (drop[i]) continue;
        new_index[i] = static_cast<long>(pruned.size());
        pruned.push_back(impl_->graph[i]);
    }
    for (long& idx : impl_->loop_factor_indices) {
        if (idx >= 0) idx = new_index[idx];
    }
    impl_->graph = pruned;
    return dropped;
}

Eigen::Isometry3d PoseGraph::Pose(int key) const {
    return FromGtsam(impl_->estimate.at<gtsam::Pose3>(gtsam::Key(key)));
}

double PoseGraph::Error() const { return impl_->graph.error(impl_->estimate); }

std::vector<double> PoseGraph::LoopChi2() const {
    std::vector<double> out;
    out.reserve(impl_->loop_factor_indices.size());
    for (long idx : impl_->loop_factor_indices) {
        if (idx < 0) {
            out.push_back(-1.0);
            continue;
        }
        const auto* bf = dynamic_cast<const gtsam::BetweenFactor<gtsam::Pose3>*>(
            impl_->graph[idx].get());
        CHECK(bf) << "loop_factor_indices points at a non-loop factor";
        // Raw Mahalanobis distance: bypass the Cauchy kernel so the number is
        // comparable against a chi2 table.
        const gtsam::Vector e = bf->evaluateError(impl_->estimate.at<gtsam::Pose3>(bf->key1()),
                                                  impl_->estimate.at<gtsam::Pose3>(bf->key2()));
        out.push_back(impl_->loop_noise_raw->squaredMahalanobisDistance(e));
    }
    return out;
}

}  // namespace pgo
}  // namespace faster_lio
