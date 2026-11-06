//
// Drives the pose graph: takes the scans in time order, seals keyframes out of them, looks for
// loops, and solves when a loop is worth it.  What it hands back is a corrected pose per scan.
//

#ifndef FASTER_LIO_PGO_BACKEND_H
#define FASTER_LIO_PGO_BACKEND_H

#include <deque>
#include <string>
#include <vector>

#include "pgo/loop_detector.h"
#include "pgo/pgo_types.h"
#include "pgo/pose_graph.h"

namespace faster_lio {
namespace pgo {

class PgoBackend {
   public:
    explicit PgoBackend(const PgoParam& prm);

    /// One scan, `body_pts` in its own frame and `pose` the odometry's T_world_body for it.  Scans
    /// must arrive in time order.  Cheap unless it seals a keyframe.
    void AddScan(double time, const Eigen::Isometry3d& pose, const std::vector<Vec3>& scan_points);

    /// Solves once more if a loop worth correcting is still waiting behind the cooldown.
    void Finish();

    const ScanNodes& nodes() const { return scan_nodes_; }
    /// The odometry pose each node came in with, before any correction.
    const Isometries& raw_poses() const { return init_poses_; }
    const LoopEdges& loops() const { return loops_; }
    int num_solves() const { return num_solves_; }

    /// loops, rejected candidates and keyframe poses, for looking at why a loop was or was not found.
    void SaveDebug(const std::string& output_dir) const;

   private:
    /// Merges the buffered scans into a keyframe anchored at `anchor_pose` and appends it; `merged`
    /// and `cells` come back in its body frame for the loop search.
    void SealKeyframe(const Eigen::Isometry3d& anchor_pose, std::vector<Vec3>* kf_points, PlaneCells* cells);
    void Solve();

    PgoParam prm_;
    PoseGraph graph_;
    LoopDetector detector_;

    ScanNodes scan_nodes_;
    Isometries init_poses_;
    KeyFrames kfs_;
    LoopEdges loops_;

    /// body-frame points of the scans waiting to be merged into the next keyframe; they are the
    /// newest stage_.size() nodes, whose poses no solve has touched since
    std::deque<std::vector<Vec3>> buf_scans_;

    /// What the last solve did to the newest node.  Every later scan comes in as dx_ * odometry, so
    /// the graph stays one rigid chain instead of a corrected past glued to an uncorrected future.
    Eigen::Isometry3d dx_ = Eigen::Isometry3d::Identity();
    double jour_ = 0.0;     ///< travelled distance [m]
    int kf_since_opt_ = 0;  ///< keyframes since the last solve, the cooldown of Voxel-SLAM
    int num_solves_ = 0;
    double unsolved_drift_ = 0.0;  ///< largest drift among loops added since the last solve
    int unsolved_loops_ = 0;       ///< loops added since the last solve
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_BACKEND_H
