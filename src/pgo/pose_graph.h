#ifndef FASTER_LIO_PGO_POSE_GRAPH_H
#define FASTER_LIO_PGO_POSE_GRAPH_H

#include <Eigen/Geometry>
#include <memory>
#include <vector>

#include "pgo/pgo_types.h"


namespace faster_lio {
namespace pgo {

/// One batch factor graph plus one set of values, solved with
/// Levenberg-Marquardt.  Nothing incremental: the whole graph is built after
/// the front end has finished.
class PoseGraph {
   public:
    explicit PoseGraph(const PgoParam& prm);
    ~PoseGraph();

    /// Anchors the gauge on keyframe 0.
    void AddPrior(int key, const Eigen::Isometry3d& pose);

    /// Odometry edge key_prev -> key_cur with the front-end relative motion.
    /// `init_cur` is the initial value of key_cur.
    void AddOdometry(int key_prev, int key_cur, const Eigen::Isometry3d& rel,
                     const Eigen::Isometry3d& init_cur);

    /// Robust (Cauchy) loop edge.
    void AddLoop(const LoopEdge& edge);

    /// Batch LM over the whole graph, warm-started from the current estimate.
    void Optimize();

    /// Drops the loop factors whose raw Mahalanobis distance at the current
    /// estimate exceeds final_outlier_chi2.  Returns how many were removed;
    /// the estimate is stale until the next Optimize().
    int RejectLoopOutliers();

    Eigen::Isometry3d Pose(int key) const;

    double Error() const;
    /// Raw (non-robustified) Mahalanobis distance of every loop factor, in the
    /// order the loops were added; -1 for a loop RejectLoopOutliers() dropped.
    std::vector<double> LoopChi2() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_POSE_GRAPH_H
