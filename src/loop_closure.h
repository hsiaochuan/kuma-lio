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

#include <memory>
#include <string>
#include <vector>

#include "eigen_type.h"
#include "local_ba.h"
#include "pgo/pgo_backend.h"

namespace kuma_lio {

// Voxel-SLAM's loop closure thread, run in step with the odometry: every scan the local BA marginalizes
// becomes a pose graph node; a solved loop moves the window, the EKF state and the trajectory by one rigid
// correction and rebuilds the local map from the newest keyframes, after which older keyframes near the
// sensor are loaded back into the map one per frame (keyframe_loading), so a revisit registers against
// the corrected old map.  Offline, so nothing is queued: a solve finishes before the next frame.
class LoopClosure {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // `prm` with keep_kf_points set; `fixed_var` is the isotropic variance of a keyframe point put back into
    // the map, as LocalBA::InitMap.
    LoopClosure(const pgo::PgoParam& prm, LocalBA* lba, double fixed_var);

    // The scan the BA just marginalized, at `time_s` (the trajectory's clock).  True when a loop was solved:
    // the BA window and the map are already corrected and `dx` is what the caller applies to its own state.
    bool AddScan(const LocalBA::MargScan& scan, double time_s, Eigen::Isometry3d* dx);
    // One history keyframe within reload_radius of `pos`, if any is left, into the local map.
    void ReloadKeyframe(const Vec3& pos);
    // At the end of the data: the scans still in the BA window (`window`, oldest first, times as AddScan's
    // from `time_offset_s`) join the graph, then the loops still waiting are solved.  Nothing is fed back:
    // the trajectory is read off backend().nodes() afterwards.
    void Finish(const std::vector<LocalBA::MargScan>& window, double time_offset_s);

    const pgo::PgoBackend& backend() const { return pgo_; }
    void SaveDebug(const std::string& dir) const;

   private:
    // Voxel-SLAM loop_update: window states, map and the keyframe bookkeeping after a solve.
    void Feedback(const Eigen::Isometry3d& dx);
    // PgoBackend::AddScan on `scan`: true when a loop worth a solve is waiting.
    bool PushScan(const LocalBA::MargScan& scan, double time_s);
    void MergeKfPoints(const pgo::KeyFrame& kf, std::vector<Vec3>* out_points) const;

    pgo::PgoParam prm_;
    LocalBA* lba_;
    pgo::PgoBackend pgo_;
    double fixed_var_;
    // One flag per history keyframe: sealed before the last solve and not among the map_kfs the map was
    // rebuilt from; set once ReloadKeyframe put it back.  Empty until the first solve.
    std::vector<bool> loaded_;
};

}  // namespace kuma_lio
