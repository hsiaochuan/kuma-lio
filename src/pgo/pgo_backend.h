#ifndef FASTER_LIO_PGO_BACKEND_H
#define FASTER_LIO_PGO_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include "pgo/pgo_types.h"
#include "stamp_pose.h"

namespace faster_lio {
namespace pgo {

class PoseGraph;
class LoopDetector;

class PgoBackend {
   public:
    explicit PgoBackend(const PgoParam& prm);
    ~PgoBackend();

    /// Called on the FRONT-END thread once per knot interval.  `world_pts` is
    /// copied; no reference into caller memory is retained (pt_meas is cleared
    /// at the top of the next ProcessMeasurement).
    void AddKnot(double t_abs, const Eigen::Isometry3d& pose_w_b, const PgoCloud& world_pts);

    /// Everything the backend computes happens here, after the front end has
    /// finished: seal the pending keyframe, build the graph, sweep for loops,
    /// optimize, reject outliers.
    void Finalize();

    /// Valid after Finalize().  Returns `traj_odom` unchanged when the backend
    /// has nothing to say (no keyframes, no loops, or a failed optimization).
    Trajectory CorrectTrajectory(const Trajectory& traj_odom) const;

    void SaveDebug(const std::string& output_dir) const;

   private:
    /// Crop, downsample and store the staging buffer as a keyframe.  Touches
    /// neither the graph nor the loop detector.
    void SealKeyFrame(double t_abs, const Eigen::Isometry3d& pose_w_b);
    /// Prior on keyframe 0 plus one odometry edge per consecutive pair.
    void BuildGraph();
    /// One pass over all keyframes; every accepted loop is added and solved.
    void DetectLoops();
    void RefreshOptimizedPoses();
    /// False when the odometry across the loop's span is already more certain
    /// than the loop measurement, so admitting it could only add error.
    bool LoopIsInformative(const LoopEdge& e) const;

    PgoParam prm_;

    // ---- front-end-thread private, no lock ----
    PgoCloudPtr staging_points_;
    Eigen::Isometry3d last_kf_pose_odom_ = Eigen::Isometry3d::Identity();
    double last_kf_time_ = 0.0;
    Eigen::Isometry3d cur_pose_odom_ = Eigen::Isometry3d::Identity();
    double cur_time_ = 0.0;
    bool have_anchor_ = false;

    // ---- graph state ----
    std::vector<KeyFrame::Ptr> kfs_;
    std::vector<LoopEdge> loops_;

    std::unique_ptr<PoseGraph> graph_;
    std::unique_ptr<LoopDetector> loop_;
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_BACKEND_H
