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

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <vector>

#include "pgo/pgo_types.h"

namespace kuma_lio {
namespace pgo {

class PoseGraph {
   public:
    explicit PoseGraph(const PgoParam& prm) : prm_(prm) {}

    // Appends node `index` at `pose`.  The first node also gets the prior; every later one gets
    // the sequential edge from its predecessor, weighted by that predecessor's `v6`.
    void AddNode(int index, const Eigen::Isometry3d& pose, const Eigen::Isometry3d& prev_pose,
                 const Vec6& prev_v6);
    void AddLoop(int from, int to, const Eigen::Isometry3d& T_from_to);

    // Solves and writes the result back over `poses`, which must hold every node added so far.
    bool Solve(Isometries* poses);

    int num_loops() const { return num_loops_; }

   private:
    PgoParam prm_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_;  // every node; after a solve, the solved poses
    int num_loops_ = 0;
};

}  // namespace pgo
}  // namespace kuma_lio
