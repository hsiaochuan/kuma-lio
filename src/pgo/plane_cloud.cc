#include "pgo/plane_cloud.h"

#include <unordered_map>

#include "common_lib.h"

namespace faster_lio {
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

PlaneCells ExtractPlaneCells(const std::vector<Vec3>& cloud, const PgoParam& prm) {
    std::unordered_map<VOXEL_LOCATION, PointCluster> cells;
    cells.reserve(cloud.size());
    // PointCluster zeroes itself on construction, so operator[] is safe here
    for (const Vec3& p : cloud) {
        if (p.allFinite()) cells[VOXEL_LOCATION(p, prm.plane_voxel)].Push(p);
    }

    PlaneCells out;
    out.reserve(cells.size());
    for (const auto& kv : cells) {
        const PointCluster& pc = kv.second;
        if (pc.N < prm.plane_min_points) continue;
        Eigen::SelfAdjointEigenSolver<Mat3> es(pc.Cov());
        if (es.eigenvalues()(0) >= prm.plane_min_eigen) continue;
        PlaneCell c;
        c.pc = pc;
        c.plane.p = pc.Mean();
        c.plane.n = es.eigenvectors().col(0);
        if (!c.plane.p.allFinite() || !c.plane.n.allFinite()) continue;
        out.push_back(c);
    }
    return out;
}

std::vector<PlanePoint> PlanePoints(const PlaneCells& cells) {
    std::vector<PlanePoint> out;
    out.reserve(cells.size());
    for (const PlaneCell& c : cells) out.push_back(c.plane);
    return out;
}

}  // namespace pgo
}  // namespace faster_lio
