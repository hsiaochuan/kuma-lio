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
#include "point_cluster.h"

namespace kuma_lio {
namespace pgo {

// A flat `plane_voxel` cell: its point statistics, which BTC merges into reference planes, and
// the centre + normal the loop ICP and the plane overlap use.
struct PlaneCell {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointCluster pc;
    PlanePoint plane;
};
using PlaneCells = std::vector<PlaneCell, Eigen::aligned_allocator<PlaneCell>>;

// Hash `cloud` into `plane_voxel` cells and keep the flat ones.
PlaneCells ExtractPlaneCells(const std::vector<Vec3>& cloud, const PgoParam& prm);

std::vector<PlanePoint> PlanePoints(const PlaneCells& cells);

std::vector<Vec3> UniformSample(const std::vector<Vec3>& cloud, double leaf);

}  // namespace pgo
}  // namespace kuma_lio
