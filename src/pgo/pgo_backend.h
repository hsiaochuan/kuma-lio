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

#pragma once

#include <deque>
#include <string>
#include <vector>

#include "pgo/loop_detector.h"
#include "pgo/pgo_types.h"
#include "pgo/pose_graph.h"

namespace kuma_lio {
namespace pgo {

class PgoBackend {
   public:
    explicit PgoBackend(const PgoParam& prm);

    // One scan, `body_pts` in its own frame and `pose` the odometry's T_world_body for it, in the frame
    // of the nodes already in (the caller's odometry must follow every Solve it asks for).  Scans must
    // arrive in time order.  Cheap unless it seals a keyframe.  Nothing is solved here: true when a loop
    // worth a solve is waiting (the paper's cooldown passed and its drift is above opt_min_drift).
    bool AddScan(double time, const Eigen::Isometry3d& pose, const std::vector<Vec3>& scan_points);
    // Solves the graph over every node and loop so far; `dx`, when given, is the rigid correction the
    // solve made at the newest node, pose_solved * pose_before^-1.  False when the solve failed.
    bool Solve(Eigen::Isometry3d* dx = nullptr);
    // Solves once more if any loop is still waiting, however small its drift (run_pgo's only solve).
    void Finish();

    const ScanNodes& nodes() const { return scan_nodes_; }
    const LoopEdges& loops() const { return loops_; }
    const KeyFrames& keyframes() const { return kfs_; }
    // Body-frame points of the scans not yet merged into a keyframe: the newest pending_scans().size() nodes.
    const std::deque<std::vector<Vec3>>& pending_scans() const { return buf_scans_; }
    int num_solves() const { return num_solves_; }

    // loops, rejected candidates and keyframe poses, for looking at why a loop was or was not found.
    void SaveDebug(const std::string& output_dir) const;

   private:
    // Merges the buffered scans into a keyframe at `scan_pose` (the newest scan) and appends it;
    // `kf_points` (its body frame) and `cell_pcs` (its plane cells' statistics) are for BTC.
    void SealKeyframe(const Eigen::Isometry3d& scan_pose, std::vector<Vec3>* kf_points,
                      std::vector<PointCluster>* cell_pcs);

    PgoParam prm_;
    PoseGraph graph_;
    LoopDetector detector_;

    ScanNodes scan_nodes_;
    KeyFrames kfs_;
    LoopEdges loops_;

    // body-frame points of the scans waiting to be merged into the next keyframe; they are the
    // newest stage_.size() nodes, whose poses no solve has touched since
    std::deque<std::vector<Vec3>> buf_scans_;

    double travel_len_ = 0.0;     // travelled distance [m]
    int kf_since_opt_ = 0;  // keyframes since the last solve, the cooldown of Voxel-SLAM
    int num_solves_ = 0;
    int unsolved_loops_ = 0;       // loops added since the last solve
};

}  // namespace pgo
}  // namespace kuma_lio
