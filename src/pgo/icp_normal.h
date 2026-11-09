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

#include <vector>

#include "pgo/pgo_types.h"

namespace kuma_lio {
namespace pgo {

struct IcpResult {
    bool ok = false;         // converged through both phases, enough matches, non-degenerate
    bool converged = false;  // reached the tight correspondence phase and settled there
    int matches = 0;
    double eigval = 0.0;  // smallest eigenvalue of sum(n n^T) over the matched normals
    int iters = 0;        // iterations actually run
    double step = 0.0;    // norm of the last translation update [m]
    bool starved = false;  // bailed out because a pass found fewer than six correspondences
    double nn_median = 0.0;  // median ungated 1-NN distance on the first pass [m]
    int src_n = 0, tar_n = 0;
    double seed_t = 0.0;  // translation of the seed transform [m]
};

// Refines `T_tar_src` so that `src` lands on `tar`, both in their own keyframe frames.
IcpResult IcpNormal(const std::vector<PlanePoint>& src, const std::vector<PlanePoint>& tar,
                    const PgoParam& prm, Eigen::Isometry3d* T_tar_src);

}  // namespace pgo
}  // namespace kuma_lio
