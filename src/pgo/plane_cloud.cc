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

#include "pgo/plane_cloud.h"

#include <unordered_map>

#include "common_lib.h"

namespace kuma_lio {
namespace pgo {

std::vector<Vec3> UniformSample(const std::vector<Vec3>& cloud, double leaf) {
    struct Best {
        size_t idx;
        double d2;
    };
    std::unordered_map<VOXEL_LOCATION, size_t> slot;  // voxel -> index in `best`
    slot.reserve(cloud.size());
    std::vector<Best> best;
    for (size_t i = 0; i < cloud.size(); ++i) {
        const Vec3& p = cloud[i];
        if (!p.allFinite()) continue;
        const VOXEL_LOCATION loc(p, leaf);
        const Vec3 center = (Vec3(loc.x, loc.y, loc.z) + Vec3::Constant(0.5)) * leaf;
        const double d2 = (p - center).squaredNorm();
        const auto ins = slot.emplace(loc, best.size());
        if (ins.second) {
            best.push_back({i, d2});
        } else if (d2 < best[ins.first->second].d2) {
            best[ins.first->second] = {i, d2};
        }
    }
    std::vector<Vec3> out;
    out.reserve(best.size());
    for (const Best& b : best) out.push_back(cloud[b.idx]);
    return out;
}

PlaneCells ExtractPlaneCells(const std::vector<Vec3>& cloud, const PgoParam& prm, std::vector<PointCluster>* cell_pcs) {
    std::unordered_map<VOXEL_LOCATION, PointCluster> cells;
    cells.reserve(cloud.size());
    // PointCluster zeroes itself on construction, so operator[] is safe here
    for (const Vec3& p : cloud) {
        if (p.allFinite()) cells[VOXEL_LOCATION(p, prm.plane_voxel)].Push(p);
    }

    PlaneCells out;
    out.reserve(cells.size());
    cell_pcs->clear();
    cell_pcs->reserve(cells.size());
    for (const auto& kv : cells) {
        const PointCluster& pc = kv.second;
        if (pc.N < prm.plane_min_points) continue;
        Eigen::SelfAdjointEigenSolver<Mat3> es(pc.Cov());
        if (es.eigenvalues()(0) >= prm.plane_min_eigen) continue;
        PlaneCell c;
        c.center = pc.Mean();
        c.normal = es.eigenvectors().col(0);
        if (!c.center.allFinite() || !c.normal.allFinite()) continue;
        out.push_back(c);
        cell_pcs->push_back(pc);
    }
    return out;
}

}  // namespace pgo
}  // namespace kuma_lio
