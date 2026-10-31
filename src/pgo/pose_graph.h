//
// The GTSAM side of the pose graph, kept behind a pimpl so no other translation unit pays for the
// GTSAM headers.  Nodes are scan poses, edges are BetweenFactor<Pose3> only: no IMU, no bias, no
// velocity, exactly as in Voxel-SLAM's loop thread.
//

#ifndef FASTER_LIO_PGO_POSE_GRAPH_H
#define FASTER_LIO_PGO_POSE_GRAPH_H

#include <memory>
#include <vector>

#include "pgo/pgo_types.h"

namespace faster_lio {
namespace pgo {

class PoseGraph {
   public:
    explicit PoseGraph(const PgoParam& prm);
    ~PoseGraph();

    /// Appends node `index` at `pose`.  The first node also gets the prior; every later one gets
    /// the sequential edge from its predecessor, weighted by that predecessor's `v6`.
    void AddNode(int index, const Eigen::Isometry3d& pose, const Eigen::Isometry3d& prev_pose,
                 const Vec6& prev_v6);
    void AddLoop(int from, int to, const Eigen::Isometry3d& T_from_to);

    /// Solves and writes the result back over `poses`, which must hold every node added so far.
    bool Solve(Isometries* poses);

    int num_loops() const { return num_loops_; }

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PgoParam prm_;
    int num_loops_ = 0;
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_POSE_GRAPH_H
